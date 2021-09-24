#command to run
#python drugprot_pytorch.py -output model_aug16_biobert -modeltype biobert -overwrite true -lr 3e-5 -epochs 1 

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
from transformers import AdamW,  get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer, AutoModel

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("-output", default=None, type=str, required=True, help="Output folder where model weights, metrics, preds will be saved")
parser.add_argument("-overwrite", default=False, type=bool, help="Set it to True to overwrite output directory")

parser.add_argument("-modeltype", default=None, type=str, help="model used [biobert,pubmedbert,bioelectra ]", required=True)
parser.add_argument("-max_seq_length", default=256, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("-do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

parser.add_argument("-train_batch_size", default=32, type=int, help="Train batch size for training.")
parser.add_argument("-valid_batch_size", default=32, type=int, help="Valid batch size for training.")
parser.add_argument("-test_batch_size", default=32, type=int, help="Test batch size for evaluation.")

parser.add_argument("-lr", default=5e-5, type=float, help="learning rate")
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
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

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
MAX_LEN = args.max_seq_length
TRAIN_BATCH_SIZE = args.train_batch_size
VALID_BATCH_SIZE = args.valid_batch_size
TEST_BATCH_SIZE = args.test_batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr
gradient_accumulation_steps = args.gradient_accumulation_steps
num_labels = 14


MODEL_CLASSES = {
    'biobert': "biobert-v1.1",
    'pubmedbert': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    'bioelectra': "kamalkraj/bioelectra-base-discriminator-pubmed",
}



cols = ["index", "sentence", "label"]
# concatenating chemprot data
df_trainchemprot = pd.read_csv("processed_chemprotdata/train.tsv", sep='\t', names=cols)
df_validchemprot = pd.read_csv("processed_chemprotdata/dev.tsv", sep='\t', names=cols)
df_testchemprot = pd.read_csv("processed_chemprotdata/test.tsv", sep='\t', names=cols)
df_chemprot = pd.concat([df_trainchemprot, df_validchemprot, df_testchemprot],  ignore_index=True)


df_traindrugprot = pd.read_csv("processed_data/train.tsv", sep='\t', names=cols)

df_train = pd.concat([df_chemprot, df_traindrugprot], ignore_index=True )
df_train = df_train.drop("index", axis=1)

# creating label map
label_list = df_train['label'].unique()
label_list.sort()
label_map = {label : i for i, label in enumerate(label_list)}
print(label_map)

# assigning numbers to labels in df_train
df_train_labels = df_train.label.values.tolist()
train_labels = []
for label in df_train_labels:
    train_labels.append(label_map[label])
df_train["label"] = train_labels

df_valid = pd.read_csv("processed_data/dev.tsv", sep='\t', names=cols)
df_valid = df_valid.drop("index", axis=1)

# assigning numbers to labels in df_valid
df_valid_labels = df_valid.label.values.tolist()
valid_labels = []
for label in df_valid_labels:
    valid_labels.append(label_map[label])
df_valid["label"] = valid_labels


#Loading test dataset 
cols = ["index", "sentence"]
df_test = pd.read_csv("processed_data/test.tsv", sep='\t', names=cols)
df_test = df_test.drop("index", axis=1)

print(df_train.head())
print(df_valid.head())
print(df_test.head())
# df_train = df_train[0:1000]
# df_valid = df_valid[0:100]
# df_test = df_test[0:100]

train_examples = df_train.shape[0]
num_train_optimization_steps = int(train_examples /TRAIN_BATCH_SIZE / gradient_accumulation_steps) * EPOCHS

config = AutoConfig.from_pretrained(MODEL_CLASSES[args.modeltype])
tokenizer = AutoTokenizer.from_pretrained(MODEL_CLASSES[args.modeltype], do_lower_case=args.do_lower_case)
model = AutoModel.from_pretrained(MODEL_CLASSES[args.modeltype], config=config)

bioelectra = False if args.modeltype != 'bioelectra' else True
n = config.hidden_size

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.sentence
        if 'label' in dataframe.columns:
            self.targets = dataframe.label
        else:
            self.targets = None
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

        if self.targets is not None:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.long)
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)   
            }


set_seed(1)
training_set = CustomDataset(df_train, tokenizer, MAX_LEN)
valid_set = CustomDataset(df_valid, tokenizer, MAX_LEN)
testing_set = CustomDataset(df_test, tokenizer, MAX_LEN)


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
test_loader = DataLoader(testing_set, **test_params)

# # Creating the customized model, by adding a drop out and a dense layer on top of bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self, model, bioelectra):
        super(BERTClass, self).__init__()
        self.l1 = model
        self.l2 = torch.nn.Dropout(0.1)
        self.l3 = torch.nn.Linear(n, num_labels)
        self.bioelectra = bioelectra
    
    def forward(self, ids, mask, token_type_ids):
        # to fix bioelectra
        if self.bioelectra:
            output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids,return_dict=False)
            output_2 = self.l2(output_1[-1][:,0])
            
        else:
            _,output_1 = self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
            output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = BERTClass(model, bioelectra)
model.to(device)

# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())


param_optimizer = list(model.named_parameters())
no_decay = ['bias','LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps)

# Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#     print(var_name, "\t", optimizer.state_dict()[var_name])

if n_gpu > 1:
    model = torch.nn.DataParallel(model)

def loss_fn(outputs, targets):
    return torch.nn.CrossEntropyLoss()(outputs, targets)

def train(epoch):
    model.train()
    running_loss = []
    #train_loss = []
    for batch_idx,data in tqdm(enumerate(training_loader, 0)):
        # import ipdb; ipdb.set_trace();
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
    avg_train_loss = sum(running_loss)/len(running_loss)
    #train_loss.append(avg_loss)
    print(f"Average training loss for Epoch : {epoch} is Loss: {avg_train_loss}")
    writer.add_scalar("Loss/train", avg_train_loss, epoch)
    return avg_train_loss


def valid(epoch):
    model.eval()
    running_loss = []
    #valid_loss = []
    for _,data in tqdm(enumerate(valid_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_fn(outputs, targets)
        running_loss.append(loss.item())
        
    avg_valid_loss = sum(running_loss)/len(running_loss)
    #valid_loss.append(avg_loss)
    print(f"Average valid loss for Epoch : {epoch} is Loss: {avg_valid_loss}")
    writer.add_scalar("Loss/valid", avg_valid_loss, epoch)
    #print("Valid loss for all epochs ", valid_loss)
    return avg_valid_loss

train_loss = []
valid_loss = []
for epoch in range(EPOCHS):
    avg_train_loss = train(epoch)
    avg_valid_loss = valid(epoch)
    train_loss.append(avg_train_loss)
    valid_loss.append(avg_valid_loss)

with open(f"{args.output}/{args.modeltype}results.txt","w") as f:
    f.write(f"Train loss for epoch : {EPOCHS} is  {str(train_loss)}")
    f.write("\n")
    f.write(f"valid loss for epoch : {EPOCHS} is  {str(valid_loss)}")

torch.save(model,os.path.join(args.output,"model.pt"))

writer.flush()

# Loading the pytorch model
loaded_model = torch.load(os.path.join(args.output, "model.pt"))

#loaded_model.load_state_dict(torch.load("/home/admin/hdd/bhuvana/biocr/drugprot/model_aug10/model.pt"))


def valid_togen_metrics():
    loaded_model.eval()
    # no targets given for the test dataset.
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(valid_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = loaded_model(ids, mask, token_type_ids)
            outputs_softmax = torch.log_softmax(outputs, dim = 1)
            outputs_cls = torch.argmax(outputs_softmax, dim = 1)   
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs_cls.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
   

print("Validation METRICS")

outputs, targets = valid_togen_metrics()
pickle.dump(outputs, open(f"{args.output}/{args.modeltype}valid_scores.pkl","wb"))

_,_,f1_score_micro,s = precision_recall_fscore_support(y_pred=outputs, 
# excluding false which is 13 in the label_map
y_true=targets, labels=[0,1,2,3,4,5,6,7,8,9,10,11,12], average="micro")

print(f"F1 Score (Micro) for validation dataset = {f1_score_micro}\n")
with open(f"{args.output}/{args.modeltype}f1score.txt","w") as file:
    file.write("\nF1 Score (Micro) " + str(f1_score_micro))
    file.write(" for learning rate " + str(LEARNING_RATE))
    file.write("\n Label map is " +str(label_map))


def test():
    loaded_model.eval()
    # no targets given for the test dataset.
    #fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(test_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            #targets = data['targets'].to(device, dtype = torch.long)
            outputs = loaded_model(ids, mask, token_type_ids)
            outputs_softmax = torch.log_softmax(outputs, dim = 1)
            outputs_cls = torch.argmax(outputs_softmax, dim = 1)   
            #fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(outputs_cls.cpu().detach().numpy().tolist())
    #return fin_outputs, fin_targets
    return fin_outputs

test_preds = test()
#print(test_preds)

pickle.dump(test_preds, open(f"{args.output}/{args.modeltype}preds.pkl","wb"))


with open(f"{args.output}/{args.modeltype}preds.txt","w") as f:
    f.write(f"{str(test_preds)}")
