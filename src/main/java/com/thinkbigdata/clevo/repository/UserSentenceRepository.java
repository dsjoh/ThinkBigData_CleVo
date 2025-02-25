package com.thinkbigdata.clevo.repository;

import com.thinkbigdata.clevo.entity.Sentence;
import com.thinkbigdata.clevo.entity.User;
import com.thinkbigdata.clevo.entity.UserSentence;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface UserSentenceRepository extends JpaRepository<UserSentence, Integer> {
    Page<UserSentence> findByUser(User user, Pageable page);
    List<UserSentence> findByUser(User user);
    Optional<UserSentence> findByUserAndSentence(User user, Sentence sentence);
}
