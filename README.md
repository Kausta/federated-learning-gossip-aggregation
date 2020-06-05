# Fully Decentralized Federated Learning with Gossip Aggregation

## Requirements

- Python 3.6+
- Following Python libraries
    - torch 1.5.0+
    - torchvision
    - tensorboardX
    - torchsummary
    - grpcio
    
```bash
PYTHONUNBUFFERED=1 nohup python main.py --id train --batch_size 128 --print_freq 100 &>../out200-classical.txt &
```
 
```bash
nohup ./run-gossip.sh <data-id>
```   
## License

Copyright © 2020 Caner Korkmaz, Arda Oztaskin