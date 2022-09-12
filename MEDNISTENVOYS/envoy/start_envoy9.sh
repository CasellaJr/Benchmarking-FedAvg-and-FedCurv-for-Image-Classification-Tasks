#!/bin/bash
set -e

fx envoy start -n env_nine --disable-tls --envoy-config-path envoy_config9.yaml -dh a40-node02 -dp 50051
