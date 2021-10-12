#!/bin/sh

tmux new-session -d -s "c-16-4-16-2" bash bin/models/coarse/crop16/16_4_16_2.sh
tmux new-session -d -s "c-16-4-16-8" bash bin/models/coarse/crop16/16_4_16_8.sh
tmux new-session -d -s "c-16-4-16-10" bash bin/models/coarse/crop16/16_4_16_10.sh
tmux new-session -d -s "c-16-4-22-2" bash bin/models/coarse/crop16/16_4_22_2.sh
tmux new-session -d -s "c-16-4-22-8" bash bin/models/coarse/crop16/16_4_22_8.sh
tmux new-session -d -s "c-16-4-22-10" bash bin/models/coarse/crop16/16_4_22_10.sh
tmux new-session -d -s "c-16-4-28-2" bash bin/models/coarse/crop16/16_4_28_2.sh
tmux new-session -d -s "c-16-4-28-8" bash bin/models/coarse/crop16/16_4_28_8.sh
tmux new-session -d -s "c-16-4-28-10" bash bin/models/coarse/crop16/16_4_28_10.sh