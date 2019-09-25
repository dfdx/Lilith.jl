# TODO: move these functions to Yota

function Yota.to_device(device::CPU, x)
    T = typeof(x)
    flds = fieldnames(T)
    if is_cuarray(x)
        return collect(x)
    elseif isempty(flds)
        # already primitive or array
        return x
    else
        # struct, recursively convert and construct type from fields
        fld_vals = [to_device(device, getfield(x, fld)) for fld in flds]
        return T(fld_vals...)
    end
end

(device::CPU)(x) = to_device(device, x)
(device::GPU)(x) = to_device(device, x)
