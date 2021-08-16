from nltk.tokenize import sent_tokenize
import os
import json

def customized_sent_tokenize(text):
    sents_raw = sent_tokenize(text)
    output_sents = []
    for sent in sents_raw:
        if len(sent.split('\t')) > 1:
            output_sents.extend(sent.split('\t'))
        else:
            output_sents.append(sent)
    return output_sents

def clean_sent(sent):
    special_chars = ['\n', '\t', '\r']
    for special_char in special_chars:
        sent = sent.replace(special_char, ' ')
    return sent


def split_sent(e1_span_s, e1_span_e, e2_span_s, e2_span_e, sent):
    # if e1 e2 not overlaping, output 5 chunks; else output 3
    pos_list = [e1_span_s, e1_span_e, e2_span_s, e2_span_e]
    if e1_span_e > e2_span_s:
        entity_s = min(e1_span_s, e2_span_s)
        entity_e = max(e1_span_e, e2_span_e)
        pos_list = [entity_s, entity_e]
    # if pos_list != sorted(pos_list):
    #     raise ValueError("Positions not in order!")
    spans = zip([0] + pos_list, pos_list + [len(sent)])
    output_chunks = []
    for (s, e) in spans:
        output_chunks.append(sent[s:e])
    return output_chunks

def dump_processed_data(output_dir, data_type, processed_data_mask):
    processed_data_mask_file = os.path.join(output_dir, f"{data_type}.tsv")
    os.makedirs(os.path.dirname(processed_data_mask_file), exist_ok=True)

    with open(processed_data_mask_file, 'w') as f:
        f.writelines(processed_data_mask)



def extract_relation_dict(relations_file, target_labels):
    # Forming relation reference dictionary
    # {doc_id:{(e1, e2): label}}
    
    with open(relations_file, 'r') as f:
        relations = f.readlines()

    relation_ref_dict = {}
    for line in relations:
        doc_id, label, e1, e2 = line.rstrip().split('\t')
        e1_id = str(e1.split(':')[1])
        e2_id = str(e2.split(':')[1])
        if doc_id not in relation_ref_dict:
            relation_ref_dict[doc_id] = {}
        label = label if label in target_labels else "false"
        relation_ref_dict[doc_id][(e1_id, e2_id)] = label
        # out_file = open("myfile.json", "w")
  
        # json.dump(relation_ref_dict, out_file, indent = 6)
  
        # out_file.close()
    
    return relation_ref_dict


def extract_entity_dict(entities_file):
    #import ipdb; ipdb.set_trace();
    # entity span refer
    # {doc_id:[[e_id, type, span_s, span_e, content]]}
    
    with open(entities_file, 'r') as f:
        entities = f.readlines()
    entity_span_dict = {}
    for line in entities:
        doc_id, e_id, type, span_s, span_e, content = line.rstrip().split('\t')
        if doc_id not in entity_span_dict:
            entity_span_dict[doc_id] = []
        # Ignoring the suffixe
        type = type.split('-')[0]
        entity_span_dict[doc_id].append(
            [e_id, type, int(span_s), int(span_e), content])
    return entity_span_dict


def reformat_data(abstract_file, relation_ref_dict, entity_span_dict):
    # Traversing abstract, and finding candidates with exact one chem
    # and one gene
    
    with open(abstract_file, 'r') as f:
        abstract_data = f.readlines()

    processed_data_mask = []
    for line in abstract_data:
        doc_id, text = line.split('\t', 1)
        sents = customized_sent_tokenize(text)
        entity_candidates = entity_span_dict[doc_id]
        prev_span_end = 0
        for sent in sents:
            # Extacting span of cur sent.
            sent_span_s = text.find(sent, prev_span_end)
            sent_span_e = sent_span_s + len(sent)
            prev_span_end = sent_span_e
            chem_list = []
            gene_list = []
            for entity_candidate in entity_candidates:
                e_id, type, entity_span_s, entity_span_e, content = \
                    entity_candidate
                if entity_span_s >= sent_span_s and entity_span_e \
                        <= sent_span_e:
                    if "CHEMICAL" in type:
                        chem_list.append(entity_candidate)
                    else:
                        gene_list.append(entity_candidate)
            #import ipdb; ipdb.set_trace();
            if len(chem_list) == 0 or len(gene_list) == 0:
                continue
            #import ipdb; ipdb.set_trace();
            # Preparing data with appending method
            for chem_candidate in chem_list:
                for gene_candidate in gene_list:

                    # Denoting the first entity entity 1.
                    if chem_candidate[2] < gene_candidate[2]:
                        e1_candidate, e2_candidate = \
                            chem_candidate, gene_candidate
                    else:
                        e2_candidate, e1_candidate = \
                            chem_candidate, gene_candidate
                    e1_id, e1_type, e1_span_s, e1_span_e, e1_content = \
                        e1_candidate
                    e2_id, e2_type, e2_span_s, e2_span_e, e2_content = \
                        e2_candidate
                    label = "false"

                    processed_doc_id = f"{doc_id}.{e1_id}.{e2_id}"

                    if doc_id in relation_ref_dict:
                        if (e1_id, e2_id) in relation_ref_dict[doc_id]:
                            label = relation_ref_dict[doc_id][(e1_id, e2_id)]
                        elif (e2_id, e1_id) in relation_ref_dict[doc_id]:
                            label = relation_ref_dict[doc_id][(e2_id, e1_id)]

                    e1_span_s, e1_span_e, e2_span_s, e2_span_e = \
                        e1_span_s - sent_span_s, e1_span_e - sent_span_s, \
                        e2_span_s - sent_span_s, e2_span_e - sent_span_s
                    # split sent into chunks
                    chunks = split_sent(
                        e1_span_s, e1_span_e, e2_span_s, e2_span_e, sent)
                    if len(chunks) == 5:
                        chunk1, chunk2_e1, chunk3, chunk4_e2, chunk5 = chunks
                        processed_sent_mask = \
                            f"{chunk1}@{e1_type}${chunk3}@{e2_type}${chunk5}"
                    else:
                        chunk1, chunk2_entity, chunk3 = chunks
                        entity_type = "CHEM-GENE"
                        processed_sent_mask = \
                            f"{chunk1}@{entity_type}${chunk3}"

                    # Forming sent using mask method
                    processed_data_mask.append(
                        f"{processed_doc_id}\t{clean_sent(processed_sent_mask)}\t{label}\n")
    return processed_data_mask



def reformat_data_test(abstract_file, entity_span_dict):
    # Traversing abstract, and finding candidates with exact one chem
    # and one gene
    
    with open(abstract_file, 'r') as f:
        abstract_data = f.readlines()

    processed_data_mask = []
    for line in abstract_data:
        doc_id, text = line.split('\t', 1)
        sents = customized_sent_tokenize(text)
        entity_candidates = entity_span_dict[doc_id]
        prev_span_end = 0
        for sent in sents:
            # Extacting span of cur sent.
            sent_span_s = text.find(sent, prev_span_end)
            sent_span_e = sent_span_s + len(sent)
            prev_span_end = sent_span_e
            chem_list = []
            gene_list = []
            for entity_candidate in entity_candidates:
                e_id, type, entity_span_s, entity_span_e, content = \
                    entity_candidate
                if entity_span_s >= sent_span_s and entity_span_e \
                        <= sent_span_e:
                    if "CHEMICAL" in type:
                        chem_list.append(entity_candidate)
                    else:
                        gene_list.append(entity_candidate)
            #import ipdb; ipdb.set_trace();
            if len(chem_list) == 0 or len(gene_list) == 0:
                continue
            #import ipdb; ipdb.set_trace();
            # Preparing data with appending method
            for chem_candidate in chem_list:
                for gene_candidate in gene_list:

                    # Denoting the first entity entity 1.
                    if chem_candidate[2] < gene_candidate[2]:
                        e1_candidate, e2_candidate = \
                            chem_candidate, gene_candidate
                    else:
                        e2_candidate, e1_candidate = \
                            chem_candidate, gene_candidate
                    e1_id, e1_type, e1_span_s, e1_span_e, e1_content = \
                        e1_candidate
                    e2_id, e2_type, e2_span_s, e2_span_e, e2_content = \
                        e2_candidate
                    #label = "false"

                    processed_doc_id = f"{doc_id}.{e1_id}.{e2_id}"

                    # if doc_id in relation_ref_dict:
                    #     if (e1_id, e2_id) in relation_ref_dict[doc_id]:
                    #         label = relation_ref_dict[doc_id][(e1_id, e2_id)]
                    #     elif (e2_id, e1_id) in relation_ref_dict[doc_id]:
                    #         label = relation_ref_dict[doc_id][(e2_id, e1_id)]

                    e1_span_s, e1_span_e, e2_span_s, e2_span_e = \
                        e1_span_s - sent_span_s, e1_span_e - sent_span_s, \
                        e2_span_s - sent_span_s, e2_span_e - sent_span_s
                    # split sent into chunks
                    chunks = split_sent(
                        e1_span_s, e1_span_e, e2_span_s, e2_span_e, sent)
                    if len(chunks) == 5:
                        chunk1, chunk2_e1, chunk3, chunk4_e2, chunk5 = chunks
                        processed_sent_mask = \
                            f"{chunk1}@{e1_type}${chunk3}@{e2_type}${chunk5}"
                    else:
                        chunk1, chunk2_entity, chunk3 = chunks
                        entity_type = "CHEM-GENE"
                        processed_sent_mask = \
                            f"{chunk1}@{entity_type}${chunk3}"

                    # Forming sent using mask method
                    processed_data_mask.append(
                        f"{processed_doc_id}\t{clean_sent(processed_sent_mask)}\n")
    return processed_data_mask

def prepare_drugprot_data(root_dir, output_dir):
    # add test data_type after test set is released.
    data_types = ['train', 'dev', 'test']
    target_labels = [ "INDIRECT-DOWNREGULATOR", "INDIRECT-UPREGULATOR", "DIRECT-REGULATOR", "ACTIVATOR", "INHIBITOR", "AGONIST", "ANTAGONIST", "AGONIST-ACTIVATOR", "AGONIST-INHIBITOR", "PRODUCT-OF", "SUBSTRATE", "SUBSTRATE_PRODUCT-OF","PART-OF"]

    # Only the training and dev dataset is provided for drug prot. When the test data is provided, 
    # change the name of the folder, prefix and affixe accoridingly below. 
    
    for data_type in data_types:
        if data_type == 'train':
            data_path = os.path.join(
                root_dir, 'training/')
            file_name_prefix = "drugprot_training_"
            file_name_affixe = ""
        elif data_type == "dev":
            data_path = os.path.join(
                root_dir, 'development/')
            file_name_prefix = "drugprot_development_"
            file_name_affixe = ""
        else:
            data_path = os.path.join(
                root_dir, 'test-background/')
            file_name_prefix = "test_background_"
            file_name_affixe = ""

        if data_type != 'test':
            relations_file = os.path.join(
                data_path, f"{file_name_prefix}relations{file_name_affixe}.tsv")
            entities_file = os.path.join(
                data_path, f"{file_name_prefix}entities{file_name_affixe}.tsv")
            abstract_file = os.path.join(
                data_path, f"{file_name_prefix}abstracts{file_name_affixe}.tsv")
            relation_ref_dict = extract_relation_dict(relations_file, target_labels)
            entity_span_dict = extract_entity_dict(entities_file)
            processed_data_mask = reformat_data(abstract_file, relation_ref_dict, entity_span_dict)
        else:
            entities_file = os.path.join(
                data_path, f"{file_name_prefix}entities{file_name_affixe}.tsv")
            abstract_file = os.path.join(
                data_path, f"{file_name_prefix}abstracts{file_name_affixe}.tsv")
            entity_span_dict = extract_entity_dict(entities_file)
            processed_data_mask = reformat_data_test(abstract_file, entity_span_dict)

        
        
        # Dumping data.
        dump_processed_data(output_dir, data_type, processed_data_mask)


if __name__ == "__main__":
    prepare_drugprot_data('drugprot-gs-training-development','processed_data')
    
