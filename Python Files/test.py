from argparse import ArgumentParser, Namespace

#parser = ArgumentParser()
#parser.add_argument('square', help="Squares a given number", type=int)
#parser.add_argument('-v', '--verbose', help="Provides a verbose description", action='store_true')

#print(parser.parse_args().v)

#main menu parser
parser = ArgumentParser(description='POC of detecting DGA domains using machine learning.')
parser.add_argument('option', type=str, choices=["prep","train","test"], help="Choose between 3 options: \nprep: Prepares the dataset\ntrain: Trains the model with the current dataset\ntest: tests the currently trained model")
args = parser.parse_args()

if args.option == "prep":
    print("prepping")
elif args.option == "train":
    print("training")
elif args.option == "test":
    print("testing")