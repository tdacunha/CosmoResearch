
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
