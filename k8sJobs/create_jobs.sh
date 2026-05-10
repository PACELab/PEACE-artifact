#!/bin/bash
cat $1 | python -c "from jinja2 import Template; import sys; print(Template(sys.stdin.read()).render());" > $2