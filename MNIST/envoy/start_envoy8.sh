#!/bin/bash
set -e

fx envoy start -n env_eight --disable-tls --envoy-config-path envoy_config8.yaml -dh a40-node01 -dp 50051

