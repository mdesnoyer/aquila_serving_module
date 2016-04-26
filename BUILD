# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-parse_headers",
        "no_layering_check",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

serving_proto_library(
    name = "aquila_inference_proto",
    srcs = ["aquila_inference.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
)

py_binary(
    name = "aquila_export",
    srcs = [
        "aquila_export.py",
    ],
    deps = [
        #”@aquila//net:aquila_model",
        "@tf//tensorflow:tensorflow_py",
        "//tensorflow_serving/session_bundle:exporter",
    ],
)

cc_binary(
    name = "aquila_inference",
    srcs = [
        "aquila_inference.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "@grpc//:grpc++",
        "@tf//tensorflow/core:framework",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:tensorflow",
        ":aquila_inference_proto",
        "//tensorflow_serving/batching:batch_scheduler",
        "//tensorflow_serving/batching:batch_scheduler_retrier",
        "//tensorflow_serving/batching:streaming_batch_scheduler",
        "//tensorflow_serving/core:manager",
        "//tensorflow_serving/core:servable_handle",
        "//tensorflow_serving/core:servable_id",
        "//tensorflow_serving/servables/tensorflow:simple_servers",
        "//tensorflow_serving/session_bundle",
        "//tensorflow_serving/session_bundle:manifest_proto",
        "//tensorflow_serving/session_bundle:signature",
        "//tensorflow_serving/util:unique_ptr_with_deps",
    ],
)

py_binary(
    name = "aquila_client",
    srcs = [
        "aquila_client.py",
        "aquila_inference_pb2.py",
    ],
    deps = ["@tf//tensorflow:tensorflow_py"],
)
