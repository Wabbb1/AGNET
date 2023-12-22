##AGNET:Adaptive feature fusion network guided by Gabor filters
##Install
```shell script
pip install -r requirement.txt
./compile.sh
```

## Training
```shell script
python train.py ${CONFIG_FILE}
```

## Test
```
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE}
```

## Speed
```shell script
python test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} --report_speed
```
