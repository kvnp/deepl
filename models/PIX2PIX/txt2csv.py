import csv

# Funktion zum Extrahieren der relevanten Informationen aus einem Zeileninhalt
def extract_info(line):
    info = {}
    parts = line.split(',')
    for part in parts:
        key, value = part.strip().split(':')
        info[key.strip()] = value.strip()
    return info

# Funktion zum Lesen der Textdatei und Schreiben der Daten in eine CSV-Datei
def text_to_csv(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'train step', 'loss d_fake', 'loss d_real', 'fake_acc', 'real_acc', 'loss fool', 'loss g']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(0, len(lines), 2):
            epoch_info = extract_info(lines[i])
            epoch_fool_info = extract_info(lines[i + 1])

            writer.writerow({
                'epoch': epoch_info.get('epoch', ''),
                'train step': epoch_info.get('train step', ''),
                'loss d_fake': epoch_info.get('loss d_fake', ''),
                'loss d_real': epoch_info.get('loss d_real', ''),
                'fake_acc': epoch_info.get('fake_acc', ''),
                'real_acc': epoch_info.get('real_acc', ''),
                'loss fool': epoch_fool_info.get('loss fool', ''),
                'loss g': epoch_fool_info.get('loss g', '')
            })

# Beispielaufruf der Funktion mit den Dateinamen
text_to_csv('train.txt', 'train.csv')
