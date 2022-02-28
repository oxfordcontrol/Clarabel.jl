function debug_cones_make_W(
    cones::ConeSet{T},
    s::ConicVector{T},
    z::ConicVector{T}
) where {T}

    W = spzeros(0,0)

    for i = 1:length(cones)
        thisW = debug_cones_make_W(cones[i],s.views[i],z.views[i])
        W = blockdiag(W,sparse(thisW))
    end

    return W
end


function debug_cones_make_W(
    K::NonnegativeCone{T},
    s::VectorView{T},
    z::VectorView{T}
) where {T}
    return sparse(Diagonal(K.w))
end


function debug_cones_make_W(
    K::SecondOrderCone{T},
    s::VectorView{T},
    z::VectorView{T}
) where {T}

    #Compute Wsoc directly from the definition in ECOS paper
    @views zbar = z ./ sqrt(z[1]^2 - norm(z[2:end])^2)
    @views sbar = s ./ sqrt(s[1]^2 - norm(s[2:end])^2)

    γ    = sqrt((1 + dot(zbar,sbar))/2)
    @views wbar = 1/2/γ .* (sbar + [zbar[1];-zbar[2:end]])
    @views η    = (s[1]^2 - norm(s[2:end])^2) / (z[1]^2 - norm(z[2:end])^2)
    η    = η^0.25
    #NB: error in ECOS paper here.   Should be 1/(1+wbar[1])
    n    = length(s)
    @views LR   = I(n-1) + (1/(1+wbar[1])).*(wbar[2:end]*wbar[2:end]')
    @views W    = [wbar[1] wbar[2:end]'; wbar[2:end] LR]
    W  .*= η
    return W

end


function debug_cones_make_WtWsparse(
    cones::ConeSet{T},
    s::ConicVector{T},
    z::ConicVector{T}
) where {T}

    W = spzeros(0,0)

    for i = 1:length(cones)
        thisW = debug_cones_make_WtWsparse(cones[i],s.views[i],z.views[i])
        W = blockdiag(W,sparse(thisW))
    end

    return W
end


function debug_cones_make_WtWsparse(
    K::NonnegativeCone{T},
    s::VectorView{T},
    z::VectorView{T}
) where {T}
    return debug_cones_make_WtW(K,s,z)
end


function debug_cones_make_WtWsparse(
    K::SecondOrderCone{T},
    s::VectorView{T},
    z::VectorView{T}
) where {T}

    D = I(length(s))*1.
    D[1] = K.d

    W = Matrix(K.η^2 .* [D K.v K.u;K.v' 1 0;K.u' 0 -1])

end




function debug_cones_make_WtW(
    cones::ConeSet{T},
    s::ConicVector{T},
    z::ConicVector{T}
) where {T}

    W = spzeros(0,0)

    for i = 1:length(cones)
        thisW = debug_cones_make_WtW(cones[i],s.views[i],z.views[i])
        W = blockdiag(W,sparse(thisW) )
    end

    return W
end


function debug_cones_make_WtW(
    K::NonnegativeCone{T},
    s::VectorView{T},
    z::VectorView{T}
) where {T}

    return sparse(Diagonal(K.w.^2))

end



function debug_cones_make_WtW(
    K::SecondOrderCone{T},
    s::VectorView{T},
    z::VectorView{T}
) where {T}

    D = I(length(s))*1.
    D[1] = K.d
    W    = K.η^2 .* (D + K.u*K.u' - K.v*K.v')

    return W

end




function debug_cones_matrixCircleOp(
    cones::ConeSet{T},
    u::ConicVector{T}
) where {T}

    L = spzeros(0,0)

    for i = 1:length(cones)
        thisL = debug_cones_matrixCircleOp(cones[i],u.views[i])
        L = blockdiag(L,sparse(thisL))
    end

    return L
end

function debug_cones_matrixCircleOp(
    K::NonnegativeCone{T},
    u::VectorView{T}
) where {T}

    return Diagonal(u)
end


function debug_cones_matrixCircleOp(
    K::SecondOrderCone{T},
    u::VectorView{T}
) where {T}

    n = length(u)
    L = [u[1] u[2:end]'; u[2:end] u[1].*I(n-1)]

    return sparse(L)
end
