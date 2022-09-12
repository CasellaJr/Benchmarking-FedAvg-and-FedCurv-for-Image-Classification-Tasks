#!/bin/bash
set -e

fx envoy start -n env_five --disable-tls --envoy-config-path envoy_config5.yaml -dh a40-node01 -dp 50051
