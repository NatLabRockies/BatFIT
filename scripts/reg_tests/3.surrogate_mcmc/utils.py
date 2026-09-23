from batfit.basicutilityc import ReadInput as ri
from batfit.model.surrogateNN import *
from batfit.utils.data_utils import *
from batfit.utils.torch_utils import *


def define_model(inp):
    """Build the surrogate from its recipe; the scalers are placeholders that
    load_state_dict fills from the checkpoint."""
    model = SurrogateFCNN(
        fc_list=inp.fc_units,
        sim_config=inp.sim_config,
        loss_fn=mae_loss,
        cyc_mode=inp.cyc_mode,
        voltage_margin=getattr(inp, "voltage_margin", 0.5),
    )
    num_parameters = get_num_parameters(model)
    print(f"No. Trainable Parameters: {num_parameters}")

    return model
