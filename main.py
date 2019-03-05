from data import process

train_path = "/home/jose/Documentos/MUIARFID/PRHLT/Autoria/data/train"
test_path = "/home/jose/Documentos/MUIARFID/PRHLT/Autoria/data/test"

print("Loading train dataset...")
dt_train = process.canon60Dataset(train_path)
print("Loaded")
print("Loading train dataset...")
dt_test = process.canon60Dataset(test_path)
print("Loaded")

