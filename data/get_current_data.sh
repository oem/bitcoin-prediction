#!/bin/bash
t="$(date +%Y-%m-%d)"
curl -s "https://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=$t"|jq  '.bpi|.[]' > all_time_close.csv