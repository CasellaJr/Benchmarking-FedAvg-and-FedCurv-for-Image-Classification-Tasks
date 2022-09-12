#!/bin/bash
set -e

fx envoy start -n env_ten --disable-tls --envoy-config-path envoy_config10.yaml -dh a40-node01 -dp 50051
