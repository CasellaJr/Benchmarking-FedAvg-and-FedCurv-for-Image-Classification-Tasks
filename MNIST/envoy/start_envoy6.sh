#!/bin/bash
set -e

fx envoy start -n env_six --disable-tls --envoy-config-path envoy_config6.yaml -dh a40-node01 -dp 50051
