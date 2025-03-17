def get_library_column_gpu(fj_of_U, test_fcns, dX, mask, D, **kwargs):
  # kwargs = {scales}
  #----------------------------------------
  # scales = scale factors ([yu], [yx], yt)
  #----------------------------------------

  # Check if scaling factors were provided
  if 'scales' in kwargs:
    (yu, yx, yt) = kwargs['scales']
    dX_ = [yx[d]*dX[d] for d in range(D)] + [yt*dX[-1]]
  else:
    dX_ = dX

  # Move tensors to GPU if available
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  fj_of_U = fj_of_U.to(device)
  test_fcns = [test_fcn.to(device) for test_fcn in test_fcns]

  # Convert test functions to appropriate PyTorch format
  conv = fj_of_U.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

  for d in range(D+1):
    # Reshape test function appropriately for 3D convolution
    test_fcn = test_fcns[d]
    slicer = [1, 1] + [1] * fj_of_U.dim()
    slicer[d+2] = -1
    test_fcn = test_fcn.view(*slicer)

    # Select appropriate convolution function
    # ADD GENERAL CASE
    if fj_of_U.dim() == 1:
      conv = nnF.conv1d(conv, test_fcn, padding='same')
    elif fj_of_U.dim() == 2:
      conv = nnF.conv2d(conv, test_fcn, padding='same')
    elif fj_of_U.dim() == 3:
      conv = nnF.conv3d(conv, test_fcn, padding='same')
    else:
      raise ValueError(f"Unsupported dimension: {fj_of_U.dim()}")

  # Remove batch and channel dimensions
  conv = conv.squeeze(0).squeeze(0)

  # Compute the scaling factor
  Lij_matrix = torch.prod(torch.tensor(dX_, device=device)) * conv
  Lij_matrix = Lij_matrix.cpu()

  # Convert to column vector over query points and return
  Lij = Lij_matrix[mask].reshape(-1, 1)

  return Lij


# Define your evaluate_fj and get_library_column functions

def parallel_worker(i, j, fj_of_U, name, test_fcns, dX, mask, D, col_kwargs):
  if (name == 'poly') and (j == 1) and (i > 1):
    return None, (i, j)
  else:
    test_fcns_i = [test_fcn[i,:] for test_fcn in test_fcns]
    Lij = get_library_column(fj_of_U, test_fcns_i, dX, mask, D, **col_kwargs)
    return Lij[:, 0], (i, j)


# Create the entire model library L

# Expects U as a list, U=[U1,...,Un]
# If aux fields exist, expects U=[...,V1,...,Vm]

def create_L_parallel(U, test_fcns, dX, mask, D, alpha, fj, **kwargs):
  # kwargs = {scales, aux_fields}
  #-------------------------------------------------
  # scales = scale factors ([yu], [yx], yt)
  # aux_fields = extra library variables [V1,...,Vm]
  #-------------------------------------------------

  #Initialize library variables
  col_kwargs = {}
  U_ = [u.clone() for u in U]

  # Check if scaling factors were provided
  if 'scales' in kwargs:
    (yu, yx, yt) = kwargs['scales']
    col_kwargs['scales'] = (yu, yx, yt)

    # Rescale each field
    for n in range(len(yu)):
      U_[n] *= yu[n]

  # Check for extra variables
  if 'aux_fields' in kwargs:
    aux_fields = kwargs['aux_fields']
  else:
    aux_fields = []
  # Note: auxiliary fields come pre-scaled
  U_ += aux_fields

  # Create function names
  fj_names = ['poly']*len(fj['poly']) + ['trig']*len(fj['trig'])

  # The library is a K x SJ matrix
  (K, S, J) = (len(mask[0]), len(alpha)-1, sum(len(fcn) for fcn in fj.values()))
  L = torch.zeros(K, S*J, dtype=U[0].dtype)

  futures = []
  with ProcessPoolExecutor() as executor:
    for j in range(1, J+1):
      name = fj_names[j-1]
      fj_of_U = evaluate_fj(name, fj, j, U_)
      for i in range(1, S+1):
        futures.append(executor.submit(parallel_worker, i, j, fj_of_U.clone(),
                                       name, test_fcns, dX, mask, D, col_kwargs))

    for future in as_completed(futures):
      result, (i, j) = future.result()
      if result is not None:
        L[:, (i-1)*J + j-1] = result

  cols_to_remove = [len(fj['poly'])*c for c in range(1, S)]
  Lib_mask = torch.ones(L.shape[1], dtype=torch.bool)
  Lib_mask[cols_to_remove] = False
  L = L[:, Lib_mask]

  return L
