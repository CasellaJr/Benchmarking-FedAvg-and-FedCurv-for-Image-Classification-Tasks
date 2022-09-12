#!/bin/bash
set -e

fx envoy start -n env_four --disable-tls --envoy-config-path envoy_config4.yaml -dh a40-node01 -dp 50051
