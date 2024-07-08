def wsindy(U, fj, alpha, **kwargs):

  # Kwarg Checking and Hyperparameters (see 'kwarg_handling')
  #---------------------------------------------------------------------------
  # Find WSINDy settings
  grid, dicts, hyperparams = kwarg_handling(U, alpha, **kwargs)

  # Individual variables/hyperparameters
  (U, D, X, dX, aux_fields) = grid
  (m, lambdas, p, tau, scales, mask) = hyperparams

  # Kwarg dictionaries
  (print_dict, tf_dict, LHS_kwargs, MSTLS_dict, model_dict) = dicts
  #---------------------------------------------------------------------------

  # Test Function Creation
  #---------------------------------------------------------------------------
  # Compute separable test functions
  test_fcns = speedy_test_fcns(m, p, tau, X, dX, alpha, D, **tf_dict)

  # LHS test functions
  LHS_tf = [test_fcn[0,:] for test_fcn in test_fcns]
  #---------------------------------------------------------------------------

  # Linear System Creation
  #---------------------------------------------------------------------------
  # Left-hand side (K x n), where n = no. of fields (U1,...,Un)
  b = []
  for n in range(len(U)):
    b.append(create_b(U[n].clone(), LHS_tf, dX, mask, D, scales=LHS_kwargs[n]))
  b = torch.cat(tuple(b), 1)

  # Model library (K x SJ), where J = beta_bar^n
  L = create_L(U, test_fcns, dX, mask, D, alpha, fj, scales=scales)
  #L = create_L_parallel(U, test_fcns, dX, mask, D, alpha, fj, scales=scales)
  #---------------------------------------------------------------------------

  # MSTLS Optimization
  #---------------------------------------------------------------------------
  # Pass precomputed w_LS as kwarg to save compute time
  w_LS = torch.linalg.lstsq(L, b, driver='gelsd').solution
  MSTLS_dict['w_LS'] = w_LS.clone()

  # Compute sparse weight vector
  w, thresh, loss_val = MSTLS(L, b, lambdas, **MSTLS_dict)
  #---------------------------------------------------------------------------

  # Print Results
  #---------------------------------------------------------------------------
  print_dict['L'] = L
  (print_dict['loss_val'], print_dict['thresh']) = (loss_val, thresh)
  print_report(**print_dict)

  # Print discovered PDE
  term_names = get_term_names(U, fj, alpha, D, **model_dict)
  pde = get_model(w, term_names)
  print(f'Discovered model:\n{pde}')
  #---------------------------------------------------------------------------

  # Return sparse weight vector
  return w
