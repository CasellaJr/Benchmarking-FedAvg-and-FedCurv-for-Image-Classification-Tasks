#!/bin/bash
set -e

fx envoy start -n env_three --disable-tls --envoy-config-path envoy_config3.yaml -dh a40-node02 -dp 50051
