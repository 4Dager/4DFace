INPUT=$1
mkdir ./aligned-placeholder
mkdir ./represent
python align-dlib.py $INPUT align outerEyesAndNose ./aligned-placeholder/ --size 96
th ./network/main.lua -outDir ./represent/ -data ./aligned-placeholder/
python classifier.py train ./represent/
