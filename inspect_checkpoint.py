import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle, json, sys
proj = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(proj, 'ray_train_outputs')
# find latest checkpoint dir
def latest_checkpoint_dir(output_dir):
    candidates = [name for name in os.listdir(output_dir) if name.startswith('checkpoint_epoch_') and os.path.isdir(os.path.join(output_dir, name))]
    if not candidates:
        return None
    return max(candidates, key=lambda n: int(n.rsplit('_',1)[-1]))

ckpt = latest_checkpoint_dir(output_dir)
if not ckpt:
    print('No checkpoint dirs found')
    sys.exit(1)
state_path = os.path.join(output_dir, ckpt, 'training_state.pkl')
print('Checkpoint dir', ckpt)
print('State path', state_path)
if not os.path.exists(state_path):
    print('State file missing')
    sys.exit(1)
with open(state_path, 'rb') as f:
    state = pickle.load(f)
print('Keys:', list(state.keys()))
print('epoch:', state.get('epoch'))
print('config epochs:', state.get('config',{}).get('epochs'))
