import os
import io
import json
import boto3
from handlers.base import BaseModel
import logging as logger
import traceback


class StreamIterator:
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line:
                self.read_pos += len(line)
                return line
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if 'chunk' not in chunk:
                print(f"Unknown event type: {chunk}")
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['chunk']['bytes'])


class model(BaseModel):
    def __init__(self, model_name, table_name):
        super().__init__(table_name)
        self.model_name = model_name
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime"
        )
        self.invoke_api = self.bedrock_client.invoke_model
        self.invoke_api_with_response_stream = self.bedrock_client.invoke_model_with_response_stream
        self.stream_iter = StreamIterator
        if self.table_name == "":
            schema_path = f'handlers/schemas/bedrock-{model_name.split(".")[0]}.json'
            if not os.path.exists(schema_path):
                raise NotImplementedError(
                    f"Schema file {schema_path} not found or not implemented."
                )
        else:
            schema_path = f'bedrock-{model_name.split(".")[0]}'
        (
            self.request_defaults,
            self.request_mapping,
            self.response_regex,
            self.response_mapping,
            self.response_stream_regex,
            self.response_stream_mapping
        ) = self.load_mappings(schema_path)


    def converse(self, model_id, messages, guardrail_config={}, inference_config={}, stream=True, system=[], tool_config={}):
        args = {
            'modelId': model_id,
            'messages': messages
        }
        if guardrail_config != {}:
            args["guardrailConfig"] = guardrail_config
        if inference_config != {}:
            args["inferenceConfig"] = inference_config
        if len(system) > 0:
            args["system"] = system
        if tool_config != {}:
            args["toolConfig"] = tool_config
        if stream:
            response = self.bedrock_client.converse_stream(**args)
            for line in self.stream_iter(response["body"]):
                if line:
                    print(f"Got line from stream: {line}")
                    output = self.parse_response(
                        line,
                        self.response_stream_mapping,
                        regex_sub=self.response_stream_regex
                    )
                yield output
        else:
            response = self.bedrock_client.converse(**args)
            print(f"Response from converse call: {response}")
            return response['output']['message']['content'][0]['text']
    
    def invoke(self, body):
        try:
            request_body = self.form_request(
                body, 
                self.request_defaults,
                self.request_mapping
            )
            response = self.invoke_api(
                modelId=self.model_name,
                body=json.dumps(request_body).encode("utf-8")
            )
            res = self.parse_response(
                response["body"].read(),
                self.response_mapping,
                regex_sub=self.response_regex
            )
            return res
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error {e}\nBody: {body}\n StackTrace: {tb}")
            return {"error": e, "stacktrace": tb}

    def invoke_with_response_stream(self, body):
        try:
            body["stream"] = True
            request_body = self.form_request(
                body, 
                self.request_defaults,
                self.request_mapping
            )
            response = self.invoke_api_with_response_stream(
                modelId=self.model_name,
                body=json.dumps(request_body).encode("utf-8")
            )
            for line in self.stream_iter(response["body"]):
                if line:
                    output = self.parse_response(
                        line,
                        self.response_stream_mapping,
                        regex_sub=self.response_stream_regex
                    )
                yield output
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error {e}\nBody: {body}\n StackTrace: {tb}")
            yield {"error": e, "stacktrace": tb}