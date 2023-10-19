# NNFS_DIY
## Custom Neural Network Framework written from scratch

>How to use

Create `Model` object
```python
import Model from NNFS_DIY
model = Model()

model.add(...)
...

model.set(
    loss=*Loss Class*,
    optimizer=*Optimizer Class*,
    accuracy=*Accuracy Class*
)

model.finalize()

model.train(*Input Data*, *Output Data*,
            validation_data=(*Test Input*, *Test Output*), 
            epochs=*Epochs*, 
            batch_size=*Items in batch*,
            print_every=*No of epoch*)

```
