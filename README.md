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
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --gossip=true --indices indices_0.pt > ../out-gossip-0.txt 2>&1 &
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --gossip=true --indices indices_1.pt > ../out-gossip-1.txt 2>&1 &
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --gossip=true --indices indices_2.pt > ../out-gossip-2.txt 2>&1 &
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --gossip=true --indices indices_3.pt > ../out-gossip-3.txt 2>&1 &
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --gossip=true --indices indices_4.pt > ../out-gossip-4.txt 2>&1 &

stdbuf -oL nohup python server.py > ../out-server.txt 2>&1 &
```   
## License

Copyright © 2020 Caner Korkmaz, Arda Oztaskin