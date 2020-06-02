# Fully Decentralized Federated Learning with Gossip Aggregation

## Requirements

- Python 3.6+
- Following Python libraries
    - torch 1.5.0+
    - torchvision
    - tensorboardX
    - torchsummary
    
```bash
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 > ../out-classical.txt 2>&1 &
```
 
```bash
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --gossip --indices indices_0 > ../out-gossip-0.txt 2>&1 &
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --gossip --indices indices_1 > ../out-gossip-1.txt 2>&1 &
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --gossip --indices indices_2 > ../out-gossip-2.txt 2>&1 &
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --gossip --indices indices_3 > ../out-gossip-3.txt 2>&1 &
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --gossip --indices indices_4 > ../out-gossip-4.txt 2>&1 &

```   
## License

Copyright © 2020 Caner Korkmaz, Arda Oztaskin