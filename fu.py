class MyDataset(Dataset):
    def __init__(self, inputTokens,outputEntity,tokenizer,max_tokens=128,mapping={'B-PER':1}):
        self.source = inputTokens
        self.target = outputEntity
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.mapping = mapping
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        
        source_data = self.source[idx[0]]
        target_data = self.target[idx[0]]
        print(target_data)
        tokenized_input = self.tokenizer(source_data, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_tokens,is_split_into_words=True)
        label_ids = torch.zeros(tokenized_input['input_ids'].size(1), dtype=torch.long)
        
        for start, end, label in target_data:
            start = int(start)
            end = int(end)
            label_id = self.mapping[label]
            label_ids[start:end+1] = label_id
        
        return tokenized_input['input_ids'].squeeze(0).to(self.device), tokenized_input['attention_mask'].squeeze(0).to(self.device), label_ids.to(self.device)
