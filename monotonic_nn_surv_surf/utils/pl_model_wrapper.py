
import pytorch_lightning as pl
import torch

def prob_mult_loss(output, target, weight):
    losses = -weight*(output*target + (1-output)*(1-target))
    return torch.mean(losses)

class LitSurvSurf(pl.LightningModule):
    def __init__(self, model, lr=0.001, print_epoch=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        # self.loss_fn = prob_mult_loss
        self.loss_fn = torch.nn.functional.binary_cross_entropy
        self.lr = lr
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.epochs_run = 0
        self.print_epoch = print_epoch
    
    def training_step(self, batch, batch_idx): #batch_idx is a compulsory argument
        xs, ts, gs, ys, weights = batch

        # Make predictions for this batch
        outputs = self.model(ts=ts, gs=gs, xs=xs)

        # Compute the loss and its gradients
        loss = self.loss_fn(outputs, ys, weight=weights)

        eval_res = dict()
        eval_res['loss'] = loss
        self.training_step_outputs.append(eval_res)        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    

    def test_step(self, batch, batch_idx): #batch_idx is a compulsory argument
        # this is the test loop
        xs, ts, gs, ys, weights = batch
        outputs = self.model(ts=ts, gs=gs, xs=xs)
        test_loss = self.loss_fn(outputs, ys, weight=weights)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)
        return {'loss':test_loss}
        

    def validation_step(self, batch, batch_idx): #batch_idx is a compulsory argument:
        # this is the validation loop
        xs, ts, gs, ys, weights = batch
        outputs = self.model(ts=ts, gs=gs, xs=xs)
        val_loss = self.loss_fn(outputs, ys, weight=weights)
        
        eval_res = dict()
        eval_res['loss'] = val_loss
        self.validation_step_outputs.append(eval_res)

    def on_train_epoch_end(self):
        self.epochs_run += 1
        train_loss = sum(output['loss'] for output in self.training_step_outputs) / len(self.training_step_outputs)
        self.log("train_loss", train_loss)

        if self.print_epoch:
            message = f'EPOCH:{self.epochs_run} training loss: {train_loss.item()} '
            print(message)
        
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        val_loss = sum(output['loss'] for output in self.validation_step_outputs) / len(self.validation_step_outputs)
        self.log("val_loss", val_loss)

        if self.print_epoch:
            message = f'EPOCH:{self.epochs_run} val loss: {val_loss.item()}'
            print(message)

        self.validation_step_outputs.clear()
