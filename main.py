from argparse import ArgumentParser
from trainer.SNGAN_trainer import trainer as SN_trainer

from models.CGAN.gen_run_script import generate
from models.CGAN.training_script import training
from utils.dataprocessor import image_processing_tool
from utils.generate_SNGAN_img import generate_img


def main():
    parser = ArgumentParser(description="Zelda SNES map generation using various GANs")
    parser.add_argument(
        '-model',
        choices=['pix2pix', 'progan', 'sngan', 'no_model'],
        help='Specify the model (pix2pix, progan, sngan)',
        required=True
    )

    parser.add_argument(
        '-method',
        choices=['train', 'test', 'generate', 'make_data'],
        help='Specify the method (train, test, generate)',
        required=True
    )

    parser.add_argument(
        '-i', '--input',
        help='Specify the input directory',
        required=False
    )
    parser.add_argument(
        '-o', '--output',
        help='Specify the output directory',
        required=True
    )

    if parser.parse_args().model == 'pix2pix':
        parser.add_argument(
            '-w', '--weights',
            help='Specify the folder containing weight files',
            required=True
        )

    parser.add_argument(
        "-f", "--fade_ins",
        help="Input 6 numbers separated with '-' or 1 number between 1 and 100",
        required=False
    )

    parser.add_argument(
        "-ep", "--epochs",
        help="Input 6 numbers separated with '-', "
             "each represents a number of epochs a resolution will get trained with",
        required=False
    )

    parser.add_argument(
        "-np", "--num_pics",
        help="Enter a number of the pictures you want to generate",
        required=False,
        type=int
    )
    
    parser.add_argument(
        "-nc", "--num_classes",
        help="Only for SNGAN, enter the amount of Classes occuring in your dataset.",
        required=False,
        type=int
    )
    
    parser.add_argument(
        "-z", "--latent_size",
        help="Sets the Latent Size of the Generator.",
        required=False,
        type=int
    )
    
    parser.add_argument(
        "-dev", "--device",
        help="The device on which the programm should run: [cpu or cuda]",
        required=False
    )

    
    
    args = parser.parse_args()
    
    print(f"Using model: {args.model}")
    print(f"Using method: {args.method}")
    if args.input:
        print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    if args.weights:
        print(f"Weights directory: {args.weights}")
    if args.fade_ins:
        print(f"Fade ins in %: {parse_fade_ins(args.fade_ins)}")
    if args.epochs:
        print(f"Epochs per resolution (1-6): {parse_epochs(args.epochs, args.model)}")
    if args.num_pics:
        print(f"Number of pictures that will be generated: {args.num_pics}")
    if args.num_classes:
        print(f"Number of Image Classes: {args.num_classes}")
    if args.latent_size:
        print(f"Latent Space Size: {args.latent_size}")
    if args.device:
        print(f"Used Device: {args.device}")

    if args.model == "progan":
        if args.method == "train":
            training(parse_epochs(args.epochs, args.model), parse_fade_ins(args.fade_ins), args.output, args.input)
        if args.method == "generate":
            generate(args.num_pics, args.input, args.output)
    
    elif args.model == "sngan":
        if args.method == "generate":
            generate_img(output_dir=args.output,
                         gen_dict=args.weights,
                         latent_size=args.latent_size,
                         num_classes=args.num_classes,
                         amount= args.num_pics,
                         device= args.device)
        elif args.method == "train":
            trn = SN_trainer(output_dir=args.output,
                     n_epoch=parse_epochs(args.epochs, args.model),
                     device=args.device,
                     class_num=args.num_classes,
                     latent_size=args.latent_size)
            trn.train() 
    
    elif args.model == "pix2pix":
        if args.method == "train":
            print("bin da")
        if args.method == "generate":
            print("bin da")
        if args.method == "test":
            print("bin da")
    
    if args.model == 'no_model' and args.method == 'make_data':
        init_data_processor(args.input, args.output)


    if args.model == 'pix2pix':
        from models.PIX2PIX.pix2pix import Pix2pix
        model = Pix2pix(args.input, args.output, args.weights)

        if args.method == 'train':
            model.train()
        elif args.method == 'generate':
            model.generate()
    
    
def init_data_processor(data_configuration, destination):
    image_processing_tool.create_dataset(config_path=data_configuration, dest_dir=destination)
  

def parse_fade_ins(fade_in: str):
    # Split the string by '-' and convert each part to an integer
    fade_ins = fade_in.split('-')
    # If there's only one number, repeat it 6 times
    if len(fade_ins) == 1:
        fade_ins = [int(fade_ins[0])] * 6
    else:
        fade_ins = [int(fade) for fade in fade_ins]
    return fade_ins


def parse_epochs(epochs, model):
    if model == "progan":
        # Split the string by '-' and convert each part to an integer
        epochs_list = epochs.split('-')
        # Ensure the list has exactly 6 numbers
        if len(epochs_list) != 6:
            raise ValueError("Epochs string must contain exactly 6 numbers.")
        epochs_list = [int(epoch) for epoch in epochs_list]
        return epochs_list
    else:
        return int(epochs)


if __name__ == "__main__":
    main()

