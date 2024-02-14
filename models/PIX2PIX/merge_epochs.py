import pandas as pd

def combine_epochs(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    data = {"epoch": []}
    for line in lines:
        columns = line.split(' ')
        epoch_pair = columns.pop(0)
        epoch = int(epoch_pair.split(':')[1])
        
        if epoch not in data["epoch"]:
            data["epoch"].append(epoch)

        for i in range(len(columns)):
            key, value = columns[i].split(':')
            key, value = (key.strip(), float(value.strip()))
            if key not in data:
                data[key] = []

            data[key].append(value)
    
    for key in data:
        print(key, len(data[key]))

    print(data["epoch"])

    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)

# Beispielaufruf der Funktion mit den Dateinamen
combine_epochs('train.txt', 'train.xlsx')
