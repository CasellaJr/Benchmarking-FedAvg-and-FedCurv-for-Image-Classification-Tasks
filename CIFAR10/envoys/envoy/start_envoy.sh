#!/bin/bash
set -e

fx envoy start -n env_one --disable-tls --envoy-config-path envoy_config.yaml -dh a40-node4 -dp 50051
