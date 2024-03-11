import sys, os, argparse
import torch


def train(
	dataset: str,
	image_encoder: str,
	text_encoder: str,
	text_decoder: str,
	batch_size: int,
	learning_rate: float,
	num_epochs: int
):
	train_set, val_set = VQADataset(dataset).split(0.9)

	train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_set, batch_size, shuffle=True)

	model = VQAModel(image_encoder, text_encoder, text_decoder)

	loss_fn = torch.nn.functional.cross_entropy_loss
	optim = torch.optim.AdamOptimizer(model.parameters(), lr=learning_rate)

	for epoch in range(num_epochs):

		for (images, questions, answers) in train_loader:
			outputs = model(images, questions)
			loss = loss_fn(outputs, answers)
			loss.backward()
			optim.step()

		for (images, questions, answers) in val_loader:
			outputs = model(images, questions)
			loss = loss_fn(outputs, answers)

	print('Done')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='VQA-RAD')
	parser.add_argument('--image_encoder', type=str, default='CLIP')
	parser.add_argument('--text_encoder', type=str, default='LLaMA')
	parser.add_argument('--text_decoder', type=str, default=None)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--learning_rate', type=float, default=1e-5)
	parser.add_argument('--num_epochs', type=int, default=100)
	kwargs = vars(parser.parse_args())
	return train(**kwargs)
