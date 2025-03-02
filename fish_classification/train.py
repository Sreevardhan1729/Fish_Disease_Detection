
import torch
import torchvision
import data_setup,model_builder,engine,utils

from torchvision import transforms

NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE=1e-3

image_dir = "data/data"

device = "cuda" if torch.cuda.is_available() else "cpu"

weights = torchvision.models.ViT_B_16_Weights.DEFAULT
transform =weights.transforms()

train_dataloader, test_dataloader, class_names = data_setup.create_data(image_dir=image_dir,transform=transform,batch_size=BATCH_SIZE)

model = model_builder.create_model(num_classes=len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters() ,lr=LEARNING_RATE)
#
engine.train(model=model,train_dataloader=train_dataloader,test_dataloader=test_dataloader,loss_fn=loss_fn,optimizer=optimizer,epochs=NUM_EPOCHS,device=device)
utils.save_model(model=model,target_dir="models",model_name="ViT.pth")