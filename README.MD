# Simple Machine Learning Config

### Advantages

- No installation / dependencies required
- Super simple: ~ 50 lines
- Retains IDE autocompletion

### Usage

Place the `config.py` file in your project root,
specify config entries as class properties,
implement the `_validate` if you want.
Then just use the config object in your training script like:

```python
# train.py

from config import Config

cfg = Config() # fails if value validation fails

lr = cfg.LEARNING_RATE # will be autocompleted by your IDE
print(lr, type(lr))

# cfg.LEARNING_RATE = 0.1 # throws exception

# cfg.dump() # dump to stdout
cfg.dump('./cfg.txt') # dump to file
```

During development (or for scripting) you can now simply
overwrite config values using an environment variable
with the same name. Types are inferred from the default
value specified in the config:

```bash
python train.py
# 0.0005 <class 'float'>

LEARNING_RATE=0.5 python train.py
# 0.5 <class 'float'>
```
