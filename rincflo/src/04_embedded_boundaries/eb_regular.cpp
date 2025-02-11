#include <rincflo.H>

void Rincflo::MakeEB_regular()
{
    EB2::AllRegularIF my_regular;
    auto gshop = EB2::makeShop(my_regular);
    EB2::Build(gshop, geom.back(), 0, 100);
}
