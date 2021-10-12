#!/bin/sh

tmux new-session -d -s "f-32-8-16-2" bash bin/models/fine/crop32/32_8_16_2.sh
tmux new-session -d -s "f-32-8-16-6" bash bin/models/fine/crop32/32_8_16_6.sh
tmux new-session -d -s "f-32-8-16-10" bash bin/models/fine/crop32/32_8_16_10.sh
tmux new-session -d -s "f-32-8-22-2" bash bin/models/fine/crop32/32_8_22_2.sh
tmux new-session -d -s "f-32-8-22-6" bash bin/models/fine/crop32/32_8_22_6.sh
tmux new-session -d -s "f-32-8-22-10" bash bin/models/fine/crop32/32_8_22_10.sh
tmux new-session -d -s "f-32-8-28-2" bash bin/models/fine/crop32/32_8_28_2.sh
tmux new-session -d -s "f-32-8-28-6" bash bin/models/fine/crop32/32_8_28_6.sh
tmux new-session -d -s "f-32-8-28-10" bash bin/models/fine/crop32/32_8_28_10.sh