import torch
import numpy as np
import BPnP
import matplotlib.pyplot as plt
import torchvision
from scipy.io import loadmat, savemat
import kornia as kn

device = 'cuda'

# Model 3d points
cube = loadmat('demo_data/cube.mat')
pts3d_gt = torch.tensor(cube['pts3d'], device=device, dtype=torch.float)
# tensor([[ 1., -1., -1.],
#         [ 1.,  1., -1.],
#         [-1.,  1., -1.],
#         [-1., -1., -1.],
#         [ 1., -1.,  1.],
#         [ 1.,  1.,  1.],
#         [-1.,  1.,  1.],
#         [-1., -1.,  1.]], device='cuda:0')
# shape: [8, 3]
n = pts3d_gt.size(0)  # 8

# Pose
poses = loadmat('demo_data/poses.mat')
# poses['poses'].shape: (50, 6)
P = torch.tensor(poses['poses'][0],
                 device=device).reshape(1, 6)  # camera poses in angle-axis
# [[-1.9496,  1.9800,  0.1222,  0.0659, -0.0374,  4.6802]]
q_gt = kn.angle_axis_to_quaternion(P[0, 0:3])  # quat

# Intrinsic Parameter
f = 300
u = 128
v = 128
K = torch.tensor([[f, 0, u], [0, f, v], [0, 0, 1]],
                 device=device,
                 dtype=torch.float)

# Projection
pts2d_gt = BPnP.batch_project(P, pts3d_gt, K)
bpnp = BPnP.BPnP.apply

# Model Setting
ite = 2000
model = torchvision.models.vgg11()
model.classifier = torch.nn.Linear(25088, n * 2)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000004)

# Init Pose
ini_pose = torch.zeros(1, 6, device=device)
ini_pose[0, 5] = 99
losses = []

# Track something
track_2d = np.empty([ite, n, 2])
track_2d_pro = np.empty([ite, n, 2])

# Realtime Ploting
plt.figure()
ax3 = plt.subplot(1, 3, 3)
plt.plot(pts2d_gt[0, :, 0].clone().detach().cpu().numpy(),
         pts2d_gt[0, :, 1].clone().detach().cpu().numpy(),
         'rs',
         ms=10.5,
         label='Target locations')
plt.title('Keypoint evolution')
ax2 = plt.subplot(1, 3, 2)
plt.plot(pts2d_gt[0, :, 0].clone().detach().cpu().numpy(),
         pts2d_gt[0, :, 1].clone().detach().cpu().numpy(),
         'rs',
         ms=10.5,
         label='Target locations')
plt.title('Pose evolution')

for i in range(ite):

    # pts2d = model(torch.ones(1, 3, 32, 32, device=device)).view(1, n, 2)
    pts2d = model(torch.ones(1, 3, 32, 32, device=device)).reshape(1, n, 2)
    track_2d[i, :, :] = pts2d.clone().cpu().detach().numpy()
    P_out = bpnp(pts2d, pts3d_gt, K, ini_pose)
    # P_out -> P_6d[i, :] = torch.cat((angle_axis, T), dim=-1)
    import pdb
    pdb.set_trace()
    pts2d_pro = BPnP.batch_project(P_out, pts3d_gt, K)

    loss = ((pts2d_pro - pts2d_gt)**2).mean() + ((pts2d_pro - pts2d)**2).mean()

    print('i: {0:4d}, loss:{1:1.9f}'.format(i, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    track_2d_pro[i, :, :] = pts2d_pro.clone().cpu().detach().numpy()

    if loss.item() < 0.001:
        break
    ini_pose = P_out.detach()
    # [[ 2.4545, -2.4925, -0.1534,  0.0661, -0.0373,  4.6799]]
    print(ini_pose)

    if i == 0:
        ax3.plot(pts2d[0, :, 0].clone().detach().cpu().numpy(),
                 pts2d[0, :, 1].clone().detach().cpu().numpy(),
                 'ko',
                 ms=8.5,
                 label='Initial location')
        ax2.plot(pts2d_pro[0, :, 0].clone().detach().cpu().numpy(),
                 pts2d_pro[0, :, 1].clone().detach().cpu().numpy(),
                 'ko',
                 ms=8.5,
                 label='Initial location')
    else:
        ax3.plot(pts2d[0, :, 0].clone().detach().cpu().numpy(),
                 pts2d[0, :, 1].clone().detach().cpu().numpy(),
                 'k.',
                 ms=0.5)
        ax2.plot(pts2d_pro[0, :, 0].clone().detach().cpu().numpy(),
                 pts2d_pro[0, :, 1].clone().detach().cpu().numpy(),
                 'k.',
                 ms=0.5)

ax3.plot(pts2d[0, :, 0].clone().detach().cpu().numpy(),
         pts2d[0, :, 1].clone().detach().cpu().numpy(),
         'go',
         ms=6,
         label='Final location')
ax2.plot(pts2d_pro[0, :, 0].clone().detach().cpu().numpy(),
         pts2d_pro[0, :, 1].clone().detach().cpu().numpy(),
         'go',
         ms=6,
         label='Final location')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.subplot(1, 3, 1)
plt.plot(list(range(len(losses))), losses)
plt.title('Loss evolution')

plt.show()

# savemat('tracks_temp.mat',{'losses':losses, 'track_2d':track_2d, 'track_2d_pro':track_2d_pro, 'pts2d_gt':pts2d_gt.cpu().numpy()})
