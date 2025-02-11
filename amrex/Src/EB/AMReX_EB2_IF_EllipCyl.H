#ifndef AMREX_EB2_IF_ELLIPCYL_H_
#define AMREX_EB2_IF_ELLIPCYL_H_
#include <AMReX_Config.H>
#include <AMReX_Array.H>
#include <AMReX_EB2_IF_Base.H>

#include <algorithm>

// For all implicit functions, >0: body; =0: boundary; <0: fluid

namespace amrex { namespace EB2 {

class EllipCylIF
    : GPUable
{
public:
    // inside: is the fluid inside the cylinder?

    //! Overloaded constructor for infinite cylinder IF: if the user does not
    //! provide a cylinder height, the IF corresponds to a cylinder that spans
    //! the whole domain.
    EllipCylIF (const RealArray& a_radii, int a_direction,
                const RealArray& a_center, bool a_inside)
        : EllipCylIF(a_radii, -1.0, a_direction, a_center, a_inside)
    { }

    //! Overloaded constructor for finite cylinder IF: if the user does specify a
    //! cylinder height (length), then the IF corresponds to a cylinder extending
    //! only to height/2 from it's center.
    EllipCylIF (const RealArray& a_radii, Real a_height, int a_direction,
                const RealArray& a_center, bool a_inside)
      : m_radii(makeXDim3(a_radii)), m_height(a_height), m_direction(a_direction),
          m_center(makeXDim3(a_center)),
          m_sign(a_inside ? 1.0 : -1.0)
        {
            AMREX_ASSERT(m_direction < AMREX_SPACEDIM);
        }

    EllipCylIF (const EllipCylIF& rhs) noexcept = default;
    EllipCylIF (EllipCylIF&& rhs) noexcept = default;
    EllipCylIF& operator= (const EllipCylIF& rhs) = delete;
    EllipCylIF& operator= (EllipCylIF&& rhs) = delete;

    AMREX_GPU_HOST_DEVICE inline
    Real operator() (AMREX_D_DECL(Real x, Real y, Real z)) const noexcept
    {
#if (AMREX_SPACEDIM == 3)
      XDim3 pos{x-m_center.x, y-m_center.y, z-m_center.z};
#else
      XDim3 pos{x-m_center.x, y-m_center.y, 0.0};
#endif
        Real d2 = 0.0;
        Real pdir;
        switch (m_direction) {
        case 0 :
        {
#if (AMREX_SPACEDIM == 3)
	  d2 = pos.y*pos.y/(m_radii.y*m_radii.y)+pos.z*pos.z/(m_radii.z*m_radii.z);
#elif (AMREX_SPACEDIM == 2)
            d2 = pos.y*pos.y/(m_radii.y*m_radii.y);
#endif
            pdir = pos.x;
            break;
        }
        case 1:
        {
#if (AMREX_SPACEDIM == 3)
            d2 = pos.x*pos.x/(m_radii.x*m_radii.x)+pos.z*pos.z/(m_radii.z*m_radii.z);
#elif (AMREX_SPACEDIM == 2)
            d2 = pos.x*pos.x/(m_radii.x*m_radii.x);
#endif
            pdir = pos.y;
            break;
        }
        default:
        {
            d2 = pos.x*pos.x/(m_radii.x*m_radii.x)+pos.y*pos.y/(m_radii.y*m_radii.y);
#if (AMREX_SPACEDIM == 3)
            pdir = pos.z;
#else
            pdir = 0.0;
#endif
            break;
        }
        }

        d2 -= 1.0;

        if (m_height < 0.0) {
            return d2*m_sign;
        } else {
            Real rtop = ( pdir - 0.5*m_height);
            Real rbot = (-pdir - 0.5*m_height);
            Real r = amrex::max(d2,rtop,rbot);
            return r*m_sign;
        }
    }

    inline Real operator() (const RealArray& p) const noexcept
    {
        return this->operator() (AMREX_D_DECL(p[0], p[1], p[2]));
    }

protected:

    XDim3     m_radii;
    Real      m_height;
    int       m_direction;
    XDim3     m_center;
    //
    Real      m_sign;
};

}}

#endif
