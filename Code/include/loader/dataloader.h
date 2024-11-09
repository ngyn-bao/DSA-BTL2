/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/*
 * File:   dataloader.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 4:01 PM
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "xtensor_lib.h"
#include "loader/dataset.h"

using namespace std;

template <typename DType, typename LType>
class DataLoader
{
public:
    class Iterator;

private:
    Dataset<DType, LType> *ptr_dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    /*TODO: add more member variables to support the iteration*/
    xt::xarray<unsigned long> indices;
    int current_index;
    int total_batches;
    int m_seed;

public:
    DataLoader(Dataset<DType, LType> *ptr_dataset,
               int batch_size,
               bool shuffle = true,
               bool drop_last = false, int seed = -1)
    {
        /*TODO: Add your code to do the initialization */
        this->ptr_dataset = ptr_dataset;
        this->batch_size = batch_size;
        this->shuffle = shuffle;
        this->drop_last = drop_last;
        this->m_seed = seed;

        int dataset_size = ptr_dataset->len();
        indices = xt::arange<unsigned long>(0, dataset_size);

        total_batches = dataset_size / batch_size;

        if (shuffle && m_seed >= 0)
        {
            xt::random::seed(m_seed);
            xt::random::shuffle(indices);
        }
        current_index = 0;
    }
    virtual ~DataLoader() {}

    // New method: from V2: begin
    int get_batch_size() { return batch_size; }
    int get_sample_count() { return ptr_dataset->len(); }
    int get_total_batch() { return int(ptr_dataset->len() / batch_size); }

    // New method: from V2: end
    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// START: Section                                                     //
    /////////////////////////////////////////////////////////////////////////

    /*TODO: Add your code here to support iteration on batch*/
public:
    class Iterator
    {
    private:
        DataLoader<DType, LType> *data_loader;
        int current_batch_index;

    public:
        Iterator(DataLoader<DType, LType> *data_loader, int current_batch_index = 0) : data_loader(data_loader), current_batch_index(current_batch_index) {}

        Batch<DType, LType> operator*()
        {
            int batch_size = data_loader->batch_size;
            int start_index = current_batch_index * batch_size;
            int dataset_len = data_loader->ptr_dataset->len();

            int end_index = min(start_index + batch_size, dataset_len);

            if (!data_loader->drop_last && (current_batch_index == data_loader->total_batches - 1))
            {
                end_index = dataset_len;
            }

            auto data_shape = data_loader->ptr_dataset->get_data_shape();
            auto label_shape = data_loader->ptr_dataset->get_label_shape();

            int current_batch_size = end_index - start_index;
            data_shape[0] = current_batch_size;

            xt::xarray<DType> batch_data = xt::zeros<DType>(data_shape);
            xt::xarray<LType> batch_label;

            if (label_shape.size() > 0)
            {
                label_shape[0] = current_batch_size;
                batch_label = xt::zeros<LType>(label_shape);
            }

            for (int i = start_index; i < end_index; i++)
            {
                int dataset_index = data_loader->indices[i];
                DataLabel<DType, LType> data_label = data_loader->ptr_dataset->getitem(dataset_index);

                xt::view(batch_data, i - start_index) = data_label.getData();

                if (label_shape.size() > 0)
                    xt::view(batch_label, i - start_index) = data_label.getLabel();
            }

            return Batch<DType, LType>(batch_data, batch_label);
        }

        Iterator &operator++()
        {
            current_batch_index++;
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator!=(const Iterator &other) const
        {
            return current_batch_index != other.current_batch_index;
        }
    };

    Iterator begin()
    {
        return Iterator(this, 0);
    }

    Iterator end()
    {
        return Iterator(this, total_batches);
    }

    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// END: Section                                                       //
    /////////////////////////////////////////////////////////////////////////
};

#endif /* DATALOADER_H */
