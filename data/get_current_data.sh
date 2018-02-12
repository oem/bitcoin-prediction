#!/bin/bash
t="$(date +%Y-%m-%d)"
curl -s "https://api.coindesk.com/v1/bpi/historical/close.json?start=2010-07-17&end=$t"|jq -r '.bpi|to_entries|.[]|[.key, .value]|@csv'|sed -e 's|"||g' > all_time_close.csv