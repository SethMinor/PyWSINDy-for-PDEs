# Artifical noise
if ('sigma_NR' in kwargs) and (kwargs['sigma_NR'] != 0.0):
  # Add Gaussian noise to the data with a given SNR
  sigma_NR = kwargs['sigma_NR']

  if multi_flag == 0: # U is a single field
    U_rms = (torch.sqrt(torch.mean(U**2))).item();
    sigma = sigma_NR * U_rms
    epsilon = torch.normal(mean=0, std=sigma, size=Ushape_, dtype=dtype_)
    U = U + epsilon

  else: # U = [U1,...,Un]
    U_rms = [(torch.sqrt(torch.mean(u**2))).item() for u in U]
    U_noise = []
    for n in range(len(U)):
      sigma = sigma_NR * U_rms[n]
      epsilon = torch.normal(mean=0, std=sigma, size=Ushape_, dtype=dtype_)
      U_noise.append(U[n] + epsilon)
    U = U_noise

  # Update the list of fields
  U_ = [[U], U][multi_flag]

  # If necessary, also apply artificial noise to aux fields [V1,...,Vm]
  if len(aux_fields) != 0:
    V_rms = [(torch.sqrt(torch.mean(v**2))).item() for v in aux_fields]
    V_noise = []
    for n in range(len(aux_fields)):
      sigma = sigma_NR * V_rms[n]
      epsilon = torch.normal(mean=0, std=sigma, size=Ushape_, dtype=dtype_)
      V_noise.append(aux_fields[n] + epsilon)
    aux_fields = V_noise
