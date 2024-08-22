print("Main1")

import util, train_utils,torch

def main():
    print("Building")
    train = SNLI("data/snli_1.0/", "train", max_data=max_data)
    val = SNLI(
        "data/snli_1.0/", "dev", max_data=max_data, vocab=(train.stoi, train.itos)
    )

    vocab_size=len(train.stoi)
    print(f"Vocab size: {vocab_size}")
    model = train_utils.build_model(
        14364,
        'bowman',
        embedding_dim=300,
        hidden_dim=512,
    )
    print("Loading")
    try:
        model.load_state_dict(torch.load("models/snli/6.pth")['load_state_dict'])
    except:
        print("Failed")
print("Main2")
main()