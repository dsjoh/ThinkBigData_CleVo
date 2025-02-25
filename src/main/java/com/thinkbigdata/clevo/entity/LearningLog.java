package com.thinkbigdata.clevo.entity;

import jakarta.persistence.*;
import lombok.*;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;
import org.springframework.format.annotation.DateTimeFormat;

import java.time.LocalDateTime;

@Entity
@Table(name = "LearningLogs")
@EntityListeners(AuditingEntityListener.class)
@Getter @Setter @Builder
@NoArgsConstructor @AllArgsConstructor
public class LearningLog {
    @Id @Column(name = "Learning_log_id") @GeneratedValue(strategy = GenerationType.AUTO)
    private Integer id;
    @JoinColumn(name = "User_id", nullable = false) @ManyToOne(fetch = FetchType.LAZY)
    private User user;
    @JoinColumn(name = "Sentence_id", nullable = false) @ManyToOne(fetch = FetchType.LAZY)
    private Sentence sentence;
    @Column(name = "Learning_log_accuracy", nullable = false)
    private Double accuracy;
    @Column(name = "Learning_log_fluency", nullable = false)
    private Double fluency;
    @Column(name = "Learning_log_vulnerable", nullable = false)
    private String vulnerable;
    @Column(name = "Learning_log_total_score", nullable = false)
    private Double totalScore;
    @CreatedDate @Column(name = "Learning_log_date", nullable = false) @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss")
    private LocalDateTime date;
}
