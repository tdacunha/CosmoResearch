
    # solve problem in abstract space:
    @tf.function()
    def eigenvalue_ode_abs_temp_1(t, y, n):
        # preprocess:
        x = tf.convert_to_tensor([y])
        # map to original space to compute Jacobian (without inversion):
        x_par = flow_P.map_to_original_coord(x)
        # precompute Jacobian and its derivative:
        jac = flow_P.inverse_jacobian(x_par)[0]
        jac_T = tf.transpose(jac)
        jac_jac_T = tf.matmul(jac, jac_T)
        # compute eigenvalues:
        eig, eigv = tf.linalg.eigh(jac_jac_T)
        w = eigv[:, n]
        #
        return w

    # solve in abstract space, propagating around the alpha equation:
    @tf.function()
    def eigenvalue_ode_abs_temp_2(t, y, n):
        # unpack y:
        x = tf.convert_to_tensor([y[:flow_P.num_params]])
        # map to original space to compute Jacobian (without inversion):
        x_par = flow_P.map_to_original_coord(x)
        # precompute Jacobian and its derivative:
        jac = flow_P.inverse_jacobian(x_par)[0]
        djac = coord_jacobian_derivative(x_par)[0]
        jacm1 = flow_P.direct_jacobian(x_par)[0]
        jac_T = tf.transpose(jac)
        jac_jac_T = tf.matmul(jac, jac_T)
        # compute eigenvalues:
        eig, eigv = tf.linalg.eigh(jac_jac_T)
        w = eigv[:, n]
        dot_J = tf.einsum('k, lk, ijl -> ji', w, jacm1, djac)
        # compute alpha equation:
        alpha_dot = 2.*tf.matmul(tf.matmul([w], jac), tf.matmul(dot_J, tf.transpose([w])))
        #
        return tf.concat([[w], alpha_dot], axis=1)[0]

    # solve full transport in abstract space:
    @tf.function()
    def eigenvalue_ode_abs_temp_3(t, y):
        # unpack y:
        x = tf.convert_to_tensor([y[:flow_P.num_params]])
        w = tf.convert_to_tensor([y[flow_P.num_params:-1]])
        alpha = tf.convert_to_tensor([y[-1]])
        # map to original space to compute Jacobian (without inversion):
        x_par = flow_P.map_to_original_coord(x)
        # precompute Jacobian and its derivative:
        jac = flow_P.inverse_jacobian(x_par)[0]
        djac = coord_jacobian_derivative(x_par)[0]
        jacm1 = flow_P.direct_jacobian(x_par)[0]
        jac_T = tf.transpose(jac)
        jac_jac_T = tf.matmul(jac, jac_T)
        I = tf.eye(flow_P.num_params)
        # select the eigenvector that we want to follow based on the solution to the continuity equation:
        eig, eigv = tf.linalg.eigh(jac_jac_T)
        idx = tf.math.argmax(tf.abs(tf.matmul(tf.transpose(eigv), tf.transpose(w))))[0]
        tilde_w =  tf.convert_to_tensor([eigv[:, idx]])
        dot_J = tf.einsum('k, lk, ijl -> ji', tilde_w[0], jacm1, djac)
        # equation for alpha:
        alpha_dot = 2.*tf.matmul(tf.matmul(tilde_w, jac), tf.matmul(dot_J, tf.transpose(tilde_w)))
        # equation for wdot:
        wdot_lhs = (jac_jac_T - tf.matmul(tf.matmul(tilde_w, jac_jac_T), tf.transpose(tilde_w))*I)
        wdot_rhs = tf.matmul(alpha_dot - tf.matmul(dot_J, jac_T) -tf.matmul(jac, tf.transpose(dot_J)), tf.transpose(tilde_w))
        w_dot = tf.linalg.lstsq(wdot_lhs, wdot_rhs, fast=False)
        w_dot = tf.matmul((I - tf.einsum('i,j->ij', tilde_w[0], tf.transpose(tilde_w[0]))), w_dot)
        # equation for w:
        x_dot = tf.transpose(tilde_w)
        #
        return tf.transpose(tf.concat([x_dot, w_dot, alpha_dot], axis=0))[0]





    # even spacing along straight lines from global PCA:
    # mode 0:
    #mode = 0
    #temp = np.array([(np.amin(P1) - y0[0]) / eigv[0, mode],
    #                 (np.amin(P2) - y0[1]) / eigv[1, mode],
    #                 (np.amax(P1) - y0[0]) / eigv[0, mode],
    #                 (np.amax(P2) - y0[1]) / eigv[1, mode]])
    #alpha_min = np.amax(temp[temp < 0])
    #alpha_max = np.amin(temp[temp > 0])
    #alpha = np.linspace(alpha_min, alpha_max, 10)
    #start_0 = np.array([y0[0]+alpha*eigv[0, mode], y0[1]+alpha*eigv[1, mode]]).astype(np.float32).T
    ## mode 1:
    #mode = 1
    #temp = np.array([(np.amin(P1) - y0[0]) / eigv[0, mode],
    #                 (np.amin(P2) - y0[1]) / eigv[1, mode],
    #                 (np.amax(P1) - y0[0]) / eigv[0, mode],
    #                 (np.amax(P2) - y0[1]) / eigv[1, mode]])
    #alpha_min = np.amax(temp[temp < 0])
    #alpha_max = np.amin(temp[temp > 0])
    #alpha = np.linspace(alpha_min, alpha_max, 10)
    #start_1 = np.array([y0[0]+alpha*eigv[0, mode], y0[1]+alpha*eigv[1, mode]]).astype(np.float32).T


















        def solve_eigenvalue_ode_abs_scipy(y0, n, length=1.5, num_points=100, **kwargs):
            """
            Solve eigenvalue ODE in abstract space, with scipy
            """
            # define solution points:
            solution_times = tf.linspace(0., length, num_points)
            # compute initial PCA:
            x_abs = tf.convert_to_tensor([y0])
            x_par = flow_P.map_to_original_coord(x_abs)
            jac = flow_P.inverse_jacobian(x_par)[0]
            jac_T = tf.transpose(jac)
            jac_jac_T = tf.matmul(jac, jac_T)
            # compute eigenvalues:
            eig, eigv = tf.linalg.eigh(jac_jac_T)
            w = eigv[:, n]
            yinit = x_abs[0]
            # solve on one side:
            #yinit = tf.concat([x_abs[0], w], axis=0)
            temp_sol_1 = scipy.integrate.solve_ivp(eigenvalue_ode,
                                                   t_span=(0.0, length),
                                                   y0=yinit,
                                                   t_eval=solution_times,
                                                   args=(n, +1.),
                                                   **kwargs)
            # solve on the other side:
            #yinit = tf.concat([x_abs[0], -w], axis=0)
            yinit = x_abs[0]
            temp_sol_2 = scipy.integrate.solve_ivp(eigenvalue_ode,
                                                   t_span=(0.0, length),
                                                   y0=yinit,
                                                   t_eval=solution_times,
                                                   args=(n, -1.),
                                                   **kwargs)
            # merge
            times = tf.concat([-temp_sol_2.t[1:][::-1], temp_sol_1.t], axis=0)
            traj = tf.concat([temp_sol_2.y[:flow_P.num_params, 1:][:, ::-1], temp_sol_1.y[:flow_P.num_params, :]], axis=1)
            vel = tf.concat([temp_sol_2.y[flow_P.num_params:, 1:][:, ::-1], temp_sol_1.y[flow_P.num_params:, :]], axis=1)
            #
            return times, traj, vel







KL_eig = np.zeros((400,2))
KL_eigv = np.zeros((400,2,2))
print(len(local_metrics))
for i in range(len(local_metrics)):

    KL_eig_i, KL_eigv_i = tf_KL_decomposition(prior_local_metrics[i], local_metrics[i])
    KL_eig[i] = KL_eig_i
    norm  = np.linalg.norm(KL_eigv_i,axis = 1)
    norm_tile = np.tile(norm,(2,1)).T
    KL_eigv[i] = KL_eigv_i/norm_tile
    # if i == 0:
    #     print(KL_eigv_i)
    #     print(norm)
    #     print(KL_eigv_i/norm_tile)
    #     print(np.linalg.norm(KL_eigv_i/norm_tile,axis = 1))
#print(np.shape(KL_eigv))
# sort PCA so first mode is index 0
