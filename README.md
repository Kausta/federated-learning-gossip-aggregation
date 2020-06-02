# Fully Decentralized Federated Learning with Gossip Aggregation

## Requirements

- Python 3.6+
- Following Python libraries
    - torch 1.5.0+
    - torchvision
    - tensorboardX
    - torchsummary
    
```bash
stdbuf -oL nohup python main.py --id train --batch_size 128 --print_freq 100 --seed 42 > ../out-classical.txt 2>&1 &
```
    
## License

Copyright © 2020 Caner Korkmaz, Arda Oztaskin