import os
import torch
import pickle
import random
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import cuda
from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW,  get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("-output", default=None, type=str, required=True, help="Output folder where model weights, metrics, preds will be saved")
parser.add_argument("-overwrite", default=False, type=bool, help="Set it to True to overwrite output directory")
parser.add_argument("-dataset", default=None, type=str, help="dataset used")
parser.add_argument("-lr", default=3e-5, type=float, help="learning rate")
parser.add_argument("-epochs", default=5, type=int, help="epochs for training")
parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. "
                            "E.g., 0.1 = 10%% of training.")
parser.add_argument("--weight_decay", default=0.01, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")

args = parser.parse_args()

if os.path.exists(args.output) and os.listdir(args.output) and not args.overwrite:
    raise ValueError("Output directory ({}) already exists and is not empty. Set the overwrite flag to overwrite".format(args.output))
if not os.path.exists(args.output):
    os.makedirs(args.output)


if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = args.epochs
LEARNING_RATE = args.lr
n = 768
gradient_accumulation_steps = 1
num_labels = 14
#import ipdb; ipdb.set_trace();
cols = ["index", "sentence", "label"]
df_train = pd.read_csv("processed_data/train.tsv", sep='\t', names=cols)
df_train = df_train.drop("index", axis=1)
LE = LabelEncoder()
df_train['label'] = LE.fit_transform(df_train.label.values)
print(LE.classes_)
pickle.dump(LE, open("chemprotLabelEncoder.pkl", 'wb'))

df_valid = pd.read_csv("processed_data/dev.tsv", sep='\t', names=cols)
df_valid = df_valid.drop("index", axis=1)
df_valid['label'] = LE.transform(df_valid.label.values)


# test set has to ber preprocessed.

# df_test = pd.read_csv("processed_data/test.tsv", sep='\t', names=cols)
# df_test = df_test.drop("index", axis=1)
# df_test['label'] = LE.transform(df_test.label.values)

print(df_train.head())
print(df_valid.head())
#print(df_test.head())

train_examples = df_train.shape[0]
num_train_optimization_steps = int(train_examples /TRAIN_BATCH_SIZE / gradient_accumulation_steps) * EPOCHS

# tokenizer = BertTokenizer.from_pretrained("biobert_v1.0_pubmed", do_lower_case=False)
# model = BertModel.from_pretrained("biobert_v1.0_pubmed")

tokenizer = BertTokenizer.from_pretrained("biobert-v1.1", do_lower_case=False)
model = BertModel.from_pretrained("biobert-v1.1")
    

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.sentence
        self.targets = dataframe.label
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding = 'max_length',
            truncation='longest_first',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.long)
        }

set_seed(1)
training_set = CustomDataset(df_train, tokenizer, MAX_LEN)
valid_set = CustomDataset(df_valid, tokenizer, MAX_LEN)
#testing_set = CustomDataset(df_test, tokenizer, MAX_LEN)

print(training_set[0])
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
valid_loader = DataLoader(valid_set, **train_params)
#test_loader = DataLoader(testing_set, **test_params)

# # Creating the customized model, by adding a drop out and a dense layer on top of bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self, model):
        super(BERTClass, self).__init__()
        self.l1 = model
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(n, num_labels)
    
    def forward(self, ids, mask, token_type_ids):
        _,output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)

        return output


model = BERTClass(model)
model.to(device)


param_optimizer = list(model.named_parameters())
no_decay = ['bias','LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

if n_gpu > 1:
    model = torch.nn.DataParallel(model)

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)


def train(epoch):
    model.train()
    running_loss = []
    train_loss = []
    for batch_idx,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        
        loss = loss_fn(outputs, targets)
        
        if n_gpu > 1:
            loss = loss.mean()        
        optimizer.zero_grad()
        loss = loss / gradient_accumulation_steps
        running_loss.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step() 
            model.zero_grad()   
    avg_loss = sum(running_loss)/len(running_loss)
    train_loss.append(avg_loss)
    print(f"Average training loss for Epoch : {epoch} is Loss: {avg_loss}")
    writer.add_scalar("Loss/train", avg_loss, epoch)


def valid(epoch):
    model.eval()
    running_loss = []
    valid_loss = []
    for _,data in tqdm(enumerate(valid_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        running_loss.append(loss.item())
        
    avg_loss = sum(running_loss)/len(running_loss)
    valid_loss.append(avg_loss)
    print(f"Average valid loss for Epoch : {epoch} is Loss: {avg_loss}")
    writer.add_scalar("Loss/valid", avg_loss, epoch)
    

for epoch in range(EPOCHS):
    train(epoch)
    valid(epoch)

writer.flush()



# def test():
#     model.eval()
#     fin_targets=[]
#     fin_outputs=[]
#     with torch.no_grad():
#         for _, data in tqdm(enumerate(test_loader, 0)):
#             ids = data['ids'].to(device, dtype = torch.long)
#             mask = data['mask'].to(device, dtype = torch.long)
#             token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#             targets = data['targets'].to(device, dtype = torch.long)
#             outputs = model(ids, mask, token_type_ids)
#             outputs_softmax = torch.log_softmax(outputs, dim = 1)
#             outputs_cls = torch.argmax(outputs_softmax, dim = 1)   
#             fin_targets.extend(targets.cpu().detach().numpy().tolist())
#             fin_outputs.extend(outputs_cls.cpu().detach().numpy().tolist())
#     return fin_outputs, fin_targets

# print("TEST METRICS")


# outputs, targets = test()


# pickle.dump(outputs, open(f"{args.output}/{args.dataset}.pkl","wb"))

# _,_,f1_score_micro,s = precision_recall_fscore_support(y_pred=outputs, y_true=targets, labels=[0,1,2,3,4], average="micro")

# print(f"F1 Score (Micro) = {f1_score_micro}\n")
# with open(f"{args.output}/{args.dataset}.txt","a") as file:
#     file.write("\nF1 Score (Micro) " + str(f1_score_micro))
#     file.write("for learning rate" + str(LEARNING_RATE))
#     file.write("\n"+str(LE.classes_))
