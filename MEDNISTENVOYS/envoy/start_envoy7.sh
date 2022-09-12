#!/bin/bash
set -e

fx envoy start -n env_seven --disable-tls --envoy-config-path envoy_config7.yaml -dh a40-node02 -dp 50051
