"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import flwr.proto.task_pb2
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class GetClientsRequest(google.protobuf.message.Message):
    """GetClients messages"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    def __init__(self,
        ) -> None: ...
global___GetClientsRequest = GetClientsRequest

class GetClientsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    def __init__(self,
        ) -> None: ...
global___GetClientsResponse = GetClientsResponse

class CreateTasksRequest(google.protobuf.message.Message):
    """CreateTasks messages"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TASK_FIELD_NUMBER: builtins.int
    @property
    def task(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[flwr.proto.task_pb2.Task]: ...
    def __init__(self,
        *,
        task: typing.Optional[typing.Iterable[flwr.proto.task_pb2.Task]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["task",b"task"]) -> None: ...
global___CreateTasksRequest = CreateTasksRequest

class CreateTasksResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TASK_IDS_FIELD_NUMBER: builtins.int
    @property
    def task_ids(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[typing.Text]: ...
    def __init__(self,
        *,
        task_ids: typing.Optional[typing.Iterable[typing.Text]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["task_ids",b"task_ids"]) -> None: ...
global___CreateTasksResponse = CreateTasksResponse

class GetResultsRequest(google.protobuf.message.Message):
    """GetResults messages"""
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    TASK_IDS_FIELD_NUMBER: builtins.int
    @property
    def task_ids(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[flwr.proto.task_pb2.Result]: ...
    def __init__(self,
        *,
        task_ids: typing.Optional[typing.Iterable[flwr.proto.task_pb2.Result]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["task_ids",b"task_ids"]) -> None: ...
global___GetResultsRequest = GetResultsRequest

class GetResultsResponse(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor
    RESULTS_FIELD_NUMBER: builtins.int
    @property
    def results(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[flwr.proto.task_pb2.Result]: ...
    def __init__(self,
        *,
        results: typing.Optional[typing.Iterable[flwr.proto.task_pb2.Result]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["results",b"results"]) -> None: ...
global___GetResultsResponse = GetResultsResponse