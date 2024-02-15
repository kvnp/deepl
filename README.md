# Nutzung
Erstellung eines Datensets
  - Für die Erstellung eines Datensets muss lediglich eine JSON datei angelegt werden, in der man notiert wird welche klasse wie viele Bilder bekommen soll.
  - Als Beispiel steht die example_config.json in utils/dataprocessor zur verfügung
  - Anschließend muss nurnoch wie folgt die main.py gestartet werden
    -python main.py -model="no_model" -method="make_data" --input="_hier den pfad zur JSON_" --output= "_hier den Pfad angeben in dem die daten enden_"

SNGAN
  - Unterstützt zwei methods: "train" und "generate"
  - Die notwendigen Parameter für *train* sind wie folgt:
    - "--output":    Der Ordner in dem die Models und sample gespeichert werden.
    - "--epochs":    Gibt die Anzahl an zu trainierenden Epochen.
    - "--device":    Das Gerät auf dem Trainiert wird [cpu oder cuda]
    - "--class_num": Gibt die Anzahl an Bildklassen an, die im Datenpool vorhanden sind.
    - "latent_size": Gibt die Größe des Latenten Raums des Generators an.

    - Input kann hier ausgelassen werden. Stattdessen sollen die Trainingsdaten in dem Ordner "data/train/train_SNGAN/" abgelegt werden. Dafür sollten die Bilder aus der vorherigen Erzeugung des Datensetzt mitsamt der Labeldatei in diesem Ordner abgelegt werden

    - Beispiel:
        python main.py -model="sngan" -method="train" --output="_gewünschter Pfad_" --epochs=20 --device="cuda" --num_classes=8 --latent_size=128

  - Die notwendigen Parameter für *generate* sind wie folgt:
    - "--output":    Der Ordner in dem die Bilder gespeichert werden.
    - "--weights":   Das State Dict eines Trainierten Generators.
    - "--num_pics":  Gibt die Anzahl der zu erzeugenden Bilder.
    - "--device":    Das Gerät auf dem Trainiert wird [cpu oder cuda]
    - "--num_classes": Gibt die Anzahl an Bildklassen an von denen erzeugt werden soll.
    - "latent_size": Gibt die Größe des Latenten Raums des Generators an.

    - Beispiel:
        python main.py -model="sngan" -method="generate" --output="_gewünschter Pfad_" --num_pics=8 --device="cuda" --num_classes=8 --latent_size=128


- ProGAN
  - ProGAN unterstüzt zwei Methoden: train und generate
  - Für beide Methoden reicht es die notwendigen Parameter zu definieren
  - Zusätzlich können weitere Parameter gesetzt werden:
    - '-f' oder '--fade_ins': 6 Zahlen oder 1 Zahl in % mit '-' separiert (50-50-50-50-50-50 oder 50)
      - definiert wie groß die Fade Ins für den Generator pro Tiefe sein werden
    - '-ep' oder '--epochs': 6 Ganzezahlen mit '-' separiert (wie oben)
      - definiert wie viele Epochen pro Tiefe das Modell trainiert wird
    - '-np' oder '--num_pics': eine Ganzezahl
      - definiert, wie viele Bilder generiert werden sollen
  - der ProGAN trainiert aufsteigend. Er beginnt mit der Auflösung 8x8 und mit jeder weiteren Tiefe steigt die Auflösung um 2^(n+1)
  - Beispiel Aufruf: python main.py -model=progan -method=train -i=./models/CGAN/ZeldaALinkToThePast-Split -o=./trained_model -ep=20-40-60-100-150-200 -f=50
  - Beispiel Ausgabe:
    
        Using model: progan
        Using method: train
        Input directory: ./models/CGAN/ZeldaALinkToThePast-Split
        Output directory: ./trained_model
        Fade ins in %: [50, 50, 50, 50, 50, 50]
        Epochs per resolution (1-6): [20, 40, 60, 100, 150, 200]
    
        Dataset was initialized
        Dataset ImageFolder
          Number of datapoints: 688
          Root location: ./models/CGAN/ZeldaALinkToThePast-Split
          StandardTransform
        Transform: ToTensor()
        [torch.Size([3, 256, 256])]
        Generator is being initialized
        Initialization complete
    
        Starting the training process ...
    
        Currently working on Depth:  1
        Current resolution: 8 x 8
        Ticker 1
    
        Epoch: 1
        Elapsed: [0:00:15.227995]  batch: 1  d_loss: -0.450853  g_loss: 3.245314


- Pix2Pix
  - Pix2Pix unterstüzt zwei Methoden: train und generate
  - Für beide Methoden sind die Argumente -i/--input (), -o/--output, -w/--weights notwendig, siehe folgende Ordner-Struktur.

    ```
    .
    ├── input
    │   ├── drawing
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── original
    │       ├── 1.png
    │       └── 2.png
    ├── output
    └── weights
        ├── best_weights.h5
        └── generator.h5

    6 directories, 6 files
    ```
    - Im übergebenen Input-Ordner erwartet das Modell die Unterordner drawing und original mit den jeweiligen Tiles des Kartensatzes und den dazugehörigen Zeichnungen.
    - Im Output-Ordner werden die Samples sowie die CSV-Dateien mit den Losses angelegt.
    - Im Weights-Ordner werden die Gewichte abgelegt und ggf. bereits vorhandene Gewichte mit den Dateinamen bes_weights.h5 und generator.h5 geladen.

  - Bei der Methode generate wird zusätzlich ein Ordner generated im Ordner output angelegt, der die Zeichnungen aus input/drawing in Kartendaten umwandelt. Der Ordner input/original ist in dem Fall nicht erforderlich.

  - Beispiel
    ```
    python main.py \
      --model pix2pix \
      --method train
      -i ./dataset \
      -o ./ \
      -w ./weights

    python main.py \
      --model pix2pix \
      --method generate
      -i ./dataset \
      -o ./ \
      -w ./weights
    ```