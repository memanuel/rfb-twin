#include <rincflo.H> 

void Rincflo::ReadRheologyParameters()
{
    #define FUNC_NAME "Rincflo::ReadRheologyParameters"
    DEBUG_FUNC_ENTRY(FUNC_NAME);

    amrex::ParmParse pp("rincflo");
    std::string fluid_model_s = "newtonian";
    pp.query("fluid_model", fluid_model_s);

    // Report fluid model if we're running the fluid simulation at all
    if (m_do_flow)
    {
        PrintStars();
        Print() << "Fluid model and rheology parameters:\n";
    }

    // Read and check parameters for the selected fluid model
    if(fluid_model_s == "newtonian")
    {
        m_fluid_model = FluidModel::Newtonian;
        if (m_do_flow)
            {Print() << format("Newtonian fluid with mu = {} .\n", m_mu);}
    }
    else if(fluid_model_s == "powerlaw")
    {
        m_fluid_model = FluidModel::powerlaw;
        pp.query("n", m_n_0);
        AMREX_ALWAYS_ASSERT(m_n_0 > 0.0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_0 != 1.0, "No point in using power-law rheology with n = 1");
        Print() << format("Power-law fluid with mu = {}, n = {}.\n", m_mu, m_n_0);
    }
    else if(fluid_model_s == "bingham")
    {
        m_fluid_model = FluidModel::Bingham;
        pp.query("tau_0", m_tau_0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tau_0 > 0.0,  "No point in using Bingham rheology with tau_0 = 0");

        pp.query("papa_reg", m_papa_reg);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_papa_reg > 0.0, "Papanastasiou regularisation parameter must be positive");
        if (m_do_flow)
        {
            Print() << format("Bingham fluid with mu = {}, tau_0 = {}, pap_reg = {}.\n", 
                m_mu, m_tau_0, m_papa_reg);
        }
    }
    else if(fluid_model_s == "hb")
    {
        m_fluid_model = FluidModel::HerschelBulkley;
        pp.query("n", m_n_0);
        AMREX_ALWAYS_ASSERT(m_n_0 > 0.0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_n_0 != 1.0, "No point in using Herschel-Bulkley rheology with n = 1");

        pp.query("tau_0", m_tau_0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tau_0 > 0.0, "No point in using Herschel-Bulkley rheology with tau_0 = 0");

        pp.query("papa_reg", m_papa_reg);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_papa_reg > 0.0, "Papanastasiou regularisation parameter must be positive");

        if (m_do_flow)
        {
            Print() << format("Herschel-Bulkley fluid with mu = {}, n = {}, tau_0 = {}, papa_reg = {}.\n", 
                m_mu, m_n_0, m_tau_0, m_papa_reg);
        }
    }
    else if(fluid_model_s == "smd")
    {
        m_fluid_model = FluidModel::deSouzaMendesDutra;
        pp.query("n", m_n_0);
        AMREX_ALWAYS_ASSERT(m_n_0 > 0.0);

        pp.query("tau_0", m_tau_0);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_tau_0 > 0.0, "No point in using de Souza Mendes-Dutra rheology with tau_0 = 0");

        pp.query("eta_0", m_eta_0);
        AMREX_ALWAYS_ASSERT(m_eta_0 > 0.0);
        if (m_do_flow)
        {
            Print() << format("de Souza Mendes-Dutra fluid with mu = {}, n = {}, tau_0 = {}, eta_0 = {}.\n",
                m_mu, m_n_0, m_tau_0, m_eta_0);
        }
    }
    else
        {Abort("Unknown fluid_model! Choose either newtonian, powerlaw, bingham, hb, smd");}

    DEBUG_FUNC_EXIT(FUNC_NAME);
    #undef FUNC_NAME
}
